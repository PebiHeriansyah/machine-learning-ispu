"""
generate_demo_data.py — Pembuat data sintetis reproducible untuk pipeline 04.

PERINGATAN: Semua data yang dihasilkan oleh skrip ini adalah SINTETIS (buatan).
Data TIDAK mewakili pengukuran ISPU resmi Provinsi Bengkulu atau lembaga manapun.
Gunakan hanya sebagai smoke test untuk memverifikasi pipeline berjalan.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def generate_synthetic_ispu(
    n_days: int = 500,
    seed: int = 42,
    station_id: str | None = None,
) -> pd.DataFrame:
    """Buat DataFrame sintetis dengan pola temporal realistis (bukan data nyata).

    Polutan mengikuti pola musiman + autoregresif sederhana. ISPU diturunkan
    dari PM25 dengan skala linear + noise. Tidak ada hubungan kausal nyata.

    Args:
        n_days: Jumlah observasi harian.
        seed: Random seed.
        station_id: Opsional, tambahkan kolom station_id.

    Returns:
        DataFrame dengan kolom: Tanggal, PM25, PM10, SO2, CO, NO2, O3,
        suhu, kelembapan, kecepatan_angin, ISPU, dan opsional station_id.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    # Pola musiman sederhana
    seasonal = 20 * np.sin(2 * np.pi * t / 365) + 10 * np.cos(4 * np.pi * t / 365)

    # PM2.5 dengan autoregresif AR(1)
    pm25 = np.zeros(n_days)
    pm25[0] = 50.0
    for i in range(1, n_days):
        pm25[i] = 0.75 * pm25[i-1] + seasonal[i] * 0.3 + rng.normal(0, 5)
    pm25 = np.clip(pm25, 5, 200)

    pm10 = pm25 * rng.uniform(1.2, 1.8, n_days) + rng.normal(0, 3, n_days)
    pm10 = np.clip(pm10, 8, 300)

    so2 = 10 + seasonal * 0.1 + rng.exponential(2, n_days)
    so2 = np.clip(so2, 0, 100)

    co = 0.5 + rng.exponential(0.3, n_days) + seasonal * 0.002
    co = np.clip(co, 0.1, 5)

    no2 = 15 + seasonal * 0.15 + rng.normal(0, 3, n_days)
    no2 = np.clip(no2, 0, 80)

    o3 = 30 - seasonal * 0.2 + rng.normal(0, 5, n_days)
    o3 = np.clip(o3, 5, 150)

    suhu = 27 + 2 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 1.5, n_days)
    suhu = np.clip(suhu, 20, 38)

    kelembapan = 75 - 5 * np.cos(2 * np.pi * t / 365) + rng.normal(0, 5, n_days)
    kelembapan = np.clip(kelembapan, 40, 99)

    kecepatan_angin = rng.exponential(2, n_days)
    kecepatan_angin = np.clip(kecepatan_angin, 0.1, 20)

    # ISPU sintetis: PM25 mendominasi, + komponen lain + noise
    ispu = (
        pm25 * 0.8
        + pm10 * 0.1
        + no2 * 0.3
        + o3 * 0.15
        + rng.normal(0, 4, n_days)
    )
    ispu = np.clip(ispu, 0, 500)

    # Sisipkan 5% nilai hilang secara acak untuk menguji imputasi
    for arr in [pm25, pm10, so2, co, no2, o3, suhu, kelembapan, kecepatan_angin]:
        mask = rng.random(n_days) < 0.05
        arr[mask] = np.nan

    df = pd.DataFrame({
        "Tanggal": dates,
        "PM25": np.round(pm25, 2),
        "PM10": np.round(pm10, 2),
        "SO2": np.round(so2, 2),
        "CO": np.round(co, 3),
        "NO2": np.round(no2, 2),
        "O3": np.round(o3, 2),
        "suhu": np.round(suhu, 1),
        "kelembapan": np.round(kelembapan, 1),
        "kecepatan_angin": np.round(kecepatan_angin, 2),
        "ISPU": np.round(ispu, 1),
    })
    if station_id:
        df.insert(1, "station_id", station_id)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buat data sintetis ISPU untuk smoke test pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-days", type=int, default=500, dest="n_days",
                        help="Jumlah baris (hari) data sintetis.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", default=str(ROOT / "data" / "synthetic_air_quality.csv"),
                        help="Path file CSV output.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_ispu(n_days=args.n_days, seed=args.seed)
    df.to_csv(output_path, index=False)
    logger.info("Data sintetis tersimpan: %s (%d baris)", output_path, len(df))
    print(
        "\n[PERINGATAN] File ini adalah DATA SINTETIS (buatan komputer).\n"
        "   Tidak mewakili pengukuran ISPU resmi Provinsi Bengkulu.\n"
        "   Gunakan hanya untuk memverifikasi bahwa pipeline berjalan.\n"
    )
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
