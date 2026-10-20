"""
pipeline.py — Pipeline utama prediksi ISPU: Random Forest baseline + ANFIS-PSO.

Judul skripsi: PEMODELAN PREDIKTIF KUALITAS UDARA MENGGUNAKAN RANDOM FOREST DAN
ADAPTIVE NEURO-FUZZY INFERENCE SYSTEM DENGAN OPTIMASI PARTICLE SWARM OPTIMIZATION
(Studi Kasus: Provinsi Bengkulu)

Alur kerja:
  1. Baca dan validasi CSV.
  2. Buat fitur lag (ISPU_lag1) dan pastikan tidak ada target leakage.
  3. Pisahkan data kronologis: train / validation / test (default 70/15/15).
  4. Imputasi dan normalisasi fitur — fit hanya pada train.
  5. Pilih fitur menggunakan Random Forest importance (fit pada train saja).
  6. Latih RF baseline pada train set.
  7. PSO mengoptimasi parameter premis ANFIS menggunakan RMSE pada validation set.
     Setiap evaluasi partikel: fit konsekuen ANFIS pada train, evaluasi pada val.
  8. Fit konsekuen ANFIS final pada train+validation.
  9. Evaluasi RF dan ANFIS-PSO satu kali pada test set.
  10. Simpan predictions.csv, metrics.json, feature_importance.csv, model artifacts.

Penggunaan:
  python src/pipeline.py                           # data demo sintetis
  python src/pipeline.py --input path/ke/data.csv  # data pengguna
  python src/pipeline.py --help                    # lihat semua opsi

CATATAN:
  Data demo sintetis hanya untuk smoke test kode. Metrik dari data sintetis tidak
  boleh dikutip sebagai hasil penelitian. Gunakan data ISPU resmi untuk skripsi.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- path setup ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anfis_pso import ANFISSugeno, PSOOptimizer  # noqa: E402

logger = logging.getLogger(__name__)

# Kolom polutan dan meteorologi yang dapat digunakan sebagai fitur
CANDIDATE_FEATURE_COLS = [
    "PM25", "PM10", "SO2", "CO", "NO2", "O3",
    "suhu", "kelembapan", "kecepatan_angin",
]
TARGET_COL = "ISPU"
DATE_COL = "Tanggal"


# ---------------------------------------------------------------------------
# Validasi dan pemuatan data
# ---------------------------------------------------------------------------

def load_and_validate(path: Path, lag: int = 1) -> pd.DataFrame:
    """Baca CSV, validasi kolom wajib, buat fitur lag, periksa leakage.

    Fitur lag ISPU (ISPU_lag{lag}) dibuat dengan menggeser kolom ISPU ke depan
    sebesar `lag` langkah waktu agar secara temporal sah digunakan sebagai prediktor.

    Args:
        path: Path ke file CSV.
        lag: Horizon prediksi (jumlah langkah waktu ke depan).

    Returns:
        DataFrame terurut, fitur lag sudah dibuat, baris NaN hasil shift dibuang.
    """
    if not path.exists():
        raise FileNotFoundError(f"File CSV tidak ditemukan: {path}")
    df = pd.read_csv(path)

    # Validasi kolom wajib
    if DATE_COL not in df.columns:
        raise ValueError(f"Kolom tanggal '{DATE_COL}' tidak ditemukan dalam CSV.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan dalam CSV.")

    # Konversi tipe
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=[DATE_COL, TARGET_COL])
    n_after = len(df)
    if n_before > n_after:
        logger.warning(
            "Dibuang %d baris karena Tanggal atau ISPU tidak valid (NaT/NaN).",
            n_before - n_after,
        )

    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Konversi fitur kandidat
    available_features = []
    for col in CANDIDATE_FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            available_features.append(col)
        else:
            logger.info("Kolom fitur '%s' tidak tersedia; akan diabaikan.", col)

    if not available_features:
        raise ValueError("Tidak ada kolom fitur polutan/meteorologi yang tersedia.")

    # Buat fitur lag ISPU (temporal, bukan leakage)
    lag_col = f"ISPU_lag{lag}"
    df[lag_col] = df[TARGET_COL].shift(lag)
    available_features.append(lag_col)

    # Periksa leakage: target pada timestamp yang sama tidak boleh menjadi fitur
    if TARGET_COL in available_features:
        raise RuntimeError(
            f"Target '{TARGET_COL}' ditemukan dalam daftar fitur — target leakage terdeteksi!"
        )

    # Buang baris NaN dari lag shift
    n_before_lag = len(df)
    df = df.dropna(subset=[lag_col]).reset_index(drop=True)
    logger.info(
        "Setelah lag shift: %d baris (buang %d baris awal karena lag).",
        len(df), n_before_lag - len(df),
    )

    df["_available_features"] = str(available_features)  # simpan sementara
    return df, available_features


# ---------------------------------------------------------------------------
# Pembagian kronologis
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bagi data secara kronologis: train / validation / test.

    Tidak ada shuffle. Periode test selalu lebih baru dari validation dan train.

    Args:
        df: DataFrame yang sudah terurut.
        train_frac: Proporsi train (0 < train_frac < 1).
        val_frac: Proporsi validation (0 < val_frac < 1; train+val < 1).

    Returns:
        (train, val, test) — masing-masing sebagai slice DataFrame.
    """
    n = len(df)
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac harus < 1.0")
    i_train = max(1, int(n * train_frac))
    i_val = max(i_train + 1, int(n * (train_frac + val_frac)))
    if i_val >= n:
        raise ValueError(
            f"Data tidak cukup untuk 3-way split dengan fraksi yang diminta "
            f"(n={n}, i_train={i_train}, i_val={i_val})."
        )
    train = df.iloc[:i_train].copy()
    val = df.iloc[i_train:i_val].copy()
    test = df.iloc[i_val:].copy()
    logger.info(
        "Split kronologis — train: %d, validation: %d, test: %d baris",
        len(train), len(val), len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Metrik evaluasi
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """Hitung MAE, RMSE, R² dengan penanganan edge case.

    Args:
        y_true: Nilai aktual.
        y_pred: Nilai prediksi.
        name: Label model untuk logging.

    Returns:
        Dict dengan kunci MAE, RMSE, R2.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if len(y_true) >= 2 and float(np.var(y_true)) > 0:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = None
        logger.warning("%s: R² tidak terdefinisi (target konstan atau < 2 sampel).", name)
    logger.info("%s | MAE=%.4f  RMSE=%.4f  R²=%s", name, mae, rmse, f"{r2:.4f}" if r2 is not None else "N/A")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:  # noqa: PLR0912, PLR0915
    """Jalankan pipeline end-to-end: load → split → RF → PSO-ANFIS → evaluasi."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Muat data ---
    input_path = Path(args.input)
    logger.info("Memuat data dari: %s", input_path)
    df, available_features = load_and_validate(input_path, lag=args.lag)
    df = df.drop(columns=["_available_features"])

    n_total = len(df)
    if n_total < 20:
        raise ValueError(
            f"Data terlalu sedikit ({n_total} baris valid). Minimal diperlukan 20 baris."
        )

    # --- 2. Split kronologis ---
    train, val, test = chronological_split(df, args.train_frac, args.val_frac)

    X_train_raw = train[available_features].values
    y_train = train[TARGET_COL].values
    X_val_raw = val[available_features].values
    y_val = val[TARGET_COL].values
    X_test_raw = test[available_features].values
    y_test = test[TARGET_COL].values

    # --- 3. Imputasi (fit pada train saja) ---
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_test_imp = imputer.transform(X_test_raw)

    # --- 4. Seleksi fitur dengan RF (fit pada train saja) ---
    rf_selector = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, random_state=args.seed, n_jobs=-1
    )
    rf_selector.fit(X_train_imp, y_train)
    importances = rf_selector.feature_importances_
    importance_df = pd.DataFrame({
        "fitur": available_features,
        "feature_importance": importances,
    }).sort_values("feature_importance", ascending=False).reset_index(drop=True)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Pilih top-k fitur untuk ANFIS
    k = min(args.anfis_features, len(available_features))
    top_features = importance_df["fitur"].iloc[:k].tolist()
    top_indices = [available_features.index(f) for f in top_features]
    n_rules = args.n_mf ** k
    logger.info(
        "Fitur ANFIS terpilih (%d fitur, %d MF/fitur = %d aturan fuzzy): %s",
        k, args.n_mf, n_rules, top_features,
    )

    X_train_top = X_train_imp[:, top_indices]
    X_val_top = X_val_imp[:, top_indices]
    X_test_top = X_test_imp[:, top_indices]

    # Normalisasi (fit pada train saja, diperlukan agar batas PSO masuk akal)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_top)
    X_val_scaled = scaler.transform(X_val_top)
    X_test_scaled = scaler.transform(X_test_top)

    # --- 5. Random Forest baseline (full feature set, fit pada train) ---
    rf_pipe = Pipeline([
        ("rf", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=args.seed, n_jobs=-1
        ))
    ])
    rf_pipe.fit(X_train_imp, y_train)
    y_pred_rf = rf_pipe.predict(X_test_imp)

    # --- 6. ANFIS-PSO ---
    anfis = ANFISSugeno(
        n_features=k,
        n_mf=args.n_mf,
        random_state=args.seed,
    )
    pso = PSOOptimizer(
        anfis=anfis,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        n_particles=args.pso_particles,
        max_iter=args.pso_iters,
        random_state=args.seed,
        patience=args.pso_patience,
    )
    logger.info("Memulai PSO (partikel=%d, max_iter=%d)...", args.pso_particles, args.pso_iters)
    best_premise = pso.optimize()
    logger.info("PSO selesai. Best validation RMSE: %.4f", pso.best_fitness)

    # Fit konsekuen ANFIS final pada train+validation
    X_trainval_scaled = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = np.concatenate([y_train, y_val])
    anfis.set_premise_flat(best_premise)
    anfis.fit_consequent(X_trainval_scaled, y_trainval)

    # Prediksi ANFIS pada test set
    y_pred_anfis = anfis.predict(X_test_scaled)

    # --- 7. Evaluasi pada test set (satu kali, tidak untuk tuning) ---
    rf_metrics = compute_metrics(y_test, y_pred_rf, "RandomForest")
    anfis_metrics = compute_metrics(y_test, y_pred_anfis, "ANFIS-PSO")

    # --- 8. Simpan hasil ---
    predictions = pd.DataFrame({
        DATE_COL: test[DATE_COL].dt.strftime("%Y-%m-%d").values,
        "ISPU_Aktual": y_test,
        "ISPU_Prediksi_RF": y_pred_rf,
        "ISPU_Prediksi_ANFIS_PSO": y_pred_anfis,
    })
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    # Simpan format yang kompatibel dengan folder 05 (dashboard)
    predictions_dashboard = pd.DataFrame({
        "Tanggal": test[DATE_COL].dt.strftime("%Y-%m-%d").values,
        "ISPU_Aktual": y_test,
        "ISPU_Prediksi": y_pred_anfis,  # default: ANFIS-PSO
        "ISPU_Prediksi_RF": y_pred_rf,
    })
    predictions_dashboard.to_csv(output_dir / "hasil_prediksi_dashboard.csv", index=False)

    metrics_out = {
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "split_strategy": f"kronologis {args.train_frac:.0%}/{args.val_frac:.0%}/{1-args.train_frac-args.val_frac:.0%}",
        "lag_horizon": args.lag,
        "anfis_features": top_features,
        "anfis_n_mf": args.n_mf,
        "anfis_n_rules": n_rules,
        "pso_n_particles": args.pso_particles,
        "pso_max_iter": args.pso_iters,
        "pso_best_val_rmse": round(float(pso.best_fitness), 6),
        "random_forest": rf_metrics,
        "anfis_pso": anfis_metrics,
        "data_note": (
            "DATA SINTETIS — bukan pengukuran ISPU resmi Provinsi Bengkulu. "
            "Metrik ini hanya membuktikan pipeline berjalan, bukan akurasi ilmiah."
            if "synthetic" in str(input_path).lower() or "sintetis" in str(input_path).lower()
            else "Ganti catatan ini dengan informasi sumber data resmi Anda."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Simpan konvergensi PSO
    pd.DataFrame({
        "iterasi": range(1, len(pso.convergence_history) + 1),
        "best_val_rmse": pso.convergence_history,
    }).to_csv(output_dir / "pso_convergence.csv", index=False)

    # Simpan model RF
    dump(rf_pipe, output_dir / "random_forest_ispu.joblib")

    # Simpan parameter ANFIS
    np.savez_compressed(
        output_dir / "anfis_params.npz",
        premise_params=anfis.premise_params,
        consequent_params=anfis.consequent_params,
        top_features=np.array(top_features),
    )

    print("\n" + "=" * 60)
    print("HASIL EVALUASI (test set kronologis)")
    print("=" * 60)
    print(json.dumps(metrics_out, indent=2, ensure_ascii=False))
    print(f"\nKeluaran tersimpan di: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline prediksi ISPU: RF baseline + ANFIS-PSO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_input = str(ROOT / "data" / "synthetic_air_quality.csv")
    default_output = str(ROOT / "output")
    parser.add_argument("--input", default=default_input, help="Path file CSV input.")
    parser.add_argument("--output", default=default_output, help="Direktori output.")
    parser.add_argument("--lag", type=int, default=1,
                        help="Horizon prediksi: jumlah langkah lag ISPU.")
    parser.add_argument("--train-frac", dest="train_frac", type=float, default=0.70,
                        help="Proporsi data training.")
    parser.add_argument("--val-frac", dest="val_frac", type=float, default=0.15,
                        help="Proporsi data validation.")
    parser.add_argument("--anfis-features", dest="anfis_features", type=int, default=3,
                        help="Jumlah fitur top-RF untuk ANFIS.")
    parser.add_argument("--n-mf", dest="n_mf", type=int, default=2,
                        help="Jumlah MF per fitur untuk ANFIS.")
    parser.add_argument("--pso-particles", dest="pso_particles", type=int, default=20,
                        help="Jumlah partikel PSO.")
    parser.add_argument("--pso-iters", dest="pso_iters", type=int, default=50,
                        help="Maksimum iterasi PSO.")
    parser.add_argument("--pso-patience", dest="pso_patience", type=int, default=10,
                        help="Berhenti awal PSO jika tidak ada perbaikan.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--verbose", action="store_true", help="Tampilkan log debug.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_pipeline(args)


if __name__ == "__main__":
    main()
