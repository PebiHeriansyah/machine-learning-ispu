"""
test_pipeline.py — Unit tests dan smoke tests untuk folder 04.

Dijalankan dengan: pytest tests/test_pipeline.py -v
atau: python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Tambahkan src ke path
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from anfis_pso import ANFISSugeno, PSOOptimizer, gaussian_mf  # noqa: E402
from generate_demo_data import generate_synthetic_ispu  # noqa: E402
from pipeline import (  # noqa: E402
    chronological_split,
    compute_metrics,
    load_and_validate,
)


# ---------------------------------------------------------------------------
# Test: Gaussian MF
# ---------------------------------------------------------------------------

class TestGaussianMF:
    def test_at_center_equals_one(self):
        c, s = 0.5, 0.2
        result = gaussian_mf(np.array([c]), c, s)
        assert abs(result[0] - 1.0) < 1e-9

    def test_output_in_unit_interval(self):
        x = np.linspace(-3, 3, 100)
        result = gaussian_mf(x, 0.0, 1.0)
        assert np.all(result >= 0) and np.all(result <= 1.0 + 1e-9)

    def test_sigma_clipped_to_minimum(self):
        # Sigma nol tidak boleh menyebabkan ZeroDivisionError
        result = gaussian_mf(np.array([0.0, 1.0]), 0.0, 0.0)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Test: ANFIS
# ---------------------------------------------------------------------------

class TestANFISSugeno:
    def test_rule_count(self):
        anfis = ANFISSugeno(n_features=3, n_mf=2)
        assert anfis.n_rules == 8  # 2^3

    def test_predict_shape(self):
        anfis = ANFISSugeno(n_features=2, n_mf=2)
        X = np.random.default_rng(0).uniform(0, 1, (20, 2))
        y = np.ones(20)
        anfis.fit_consequent(X, y)
        pred = anfis.predict(X)
        assert pred.shape == (20,)

    def test_fit_consequent_and_predict_finite(self):
        rng = np.random.default_rng(99)
        X = rng.uniform(-1, 1, (50, 3))
        y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + rng.normal(0, 0.1, 50)
        anfis = ANFISSugeno(n_features=3, n_mf=2, random_state=0)
        anfis.fit_consequent(X, y)
        pred = anfis.predict(X)
        assert np.all(np.isfinite(pred)), "Prediksi ANFIS harus terbatas (finite)"

    def test_get_set_premise_flat_roundtrip(self):
        anfis = ANFISSugeno(n_features=2, n_mf=2)
        flat = anfis.get_premise_flat()
        anfis.set_premise_flat(flat * 1.5)
        retrieved = anfis.get_premise_flat()
        np.testing.assert_allclose(retrieved, flat * 1.5)

    def test_wrong_premise_length_raises(self):
        anfis = ANFISSugeno(n_features=2, n_mf=2)
        with pytest.raises(ValueError, match="Panjang vektor premis"):
            anfis.set_premise_flat(np.zeros(5))  # harus 2*2*2=8


# ---------------------------------------------------------------------------
# Test: PSO
# ---------------------------------------------------------------------------

class TestPSOOptimizer:
    def test_pso_returns_valid_params(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 2))
        y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.2, 60)
        anfis = ANFISSugeno(n_features=2, n_mf=2, random_state=0)
        pso = PSOOptimizer(
            anfis=anfis,
            X_train=X[:40], y_train=y[:40],
            X_val=X[40:], y_val=y[40:],
            n_particles=5, max_iter=5, random_state=0
        )
        best = pso.optimize()
        # dim = n_features * n_mf * 2 = 2 * 2 * 2 = 8
        expected_dim = anfis.n_features * anfis.n_mf * 2
        assert best.shape == (expected_dim,)
        assert np.all(np.isfinite(best))
        assert pso.best_fitness < float("inf"), "PSO harus menemukan solusi yang valid"

    def test_pso_convergence_history_length(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 2))
        y = rng.standard_normal(40)
        anfis = ANFISSugeno(n_features=2, n_mf=2)
        pso = PSOOptimizer(
            anfis=anfis,
            X_train=X[:30], y_train=y[:30],
            X_val=X[30:], y_val=y[30:],
            n_particles=3, max_iter=4, patience=10
        )
        pso.optimize()
        assert len(pso.convergence_history) <= 4  # tidak melebihi max_iter


# ---------------------------------------------------------------------------
# Test: Chronological split
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    def _make_df(self, n=100):
        dates = pd.date_range("2022-01-01", periods=n)
        return pd.DataFrame({"Tanggal": dates, "ISPU": np.arange(n, dtype=float)})

    def test_train_before_val_before_test(self):
        df = self._make_df(100)
        train, val, test = chronological_split(df, 0.70, 0.15)
        assert train["Tanggal"].max() < val["Tanggal"].min(), \
            "Semua tanggal train harus lebih awal dari tanggal validation"
        assert val["Tanggal"].max() < test["Tanggal"].min(), \
            "Semua tanggal validation harus lebih awal dari tanggal test"

    def test_no_overlap(self):
        df = self._make_df(100)
        train, val, test = chronological_split(df, 0.70, 0.15)
        train_dates = set(train["Tanggal"])
        val_dates = set(val["Tanggal"])
        test_dates = set(test["Tanggal"])
        assert len(train_dates & val_dates) == 0, "Tidak boleh ada tumpang-tindih train/val"
        assert len(val_dates & test_dates) == 0, "Tidak boleh ada tumpang-tindih val/test"
        assert len(train_dates & test_dates) == 0, "Tidak boleh ada tumpang-tindih train/test"

    def test_total_rows(self):
        df = self._make_df(100)
        train, val, test = chronological_split(df, 0.70, 0.15)
        assert len(train) + len(val) + len(test) == 100

    def test_invalid_frac_raises(self):
        df = self._make_df(50)
        with pytest.raises(ValueError):
            chronological_split(df, 0.80, 0.25)  # sum > 1


# ---------------------------------------------------------------------------
# Test: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_metrics(y, y, "test")
        assert m["MAE"] == 0.0
        assert m["RMSE"] == 0.0
        assert abs(m["R2"] - 1.0) < 1e-9

    def test_constant_target_r2_none(self):
        y = np.array([5.0, 5.0, 5.0])
        m = compute_metrics(y, y + 0.1, "test")
        assert m["R2"] is None, "R² harus None jika target konstan"

    def test_single_sample_r2_none(self):
        y = np.array([3.0])
        m = compute_metrics(y, y + 1.0, "test")
        assert m["R2"] is None, "R² harus None jika hanya 1 sampel"


# ---------------------------------------------------------------------------
# Test: load_and_validate (smoke test dengan file sementara)
# ---------------------------------------------------------------------------

class TestLoadAndValidate:
    def test_valid_csv(self):
        df_demo = generate_synthetic_ispu(n_days=60, seed=0)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            df_demo.to_csv(f, index=False)
            tmp_path = Path(f.name)
        try:
            result, features = load_and_validate(tmp_path, lag=1)
            assert "ISPU_lag1" in features
            assert len(result) <= 60  # beberapa baris mungkin dibuang karena lag
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_missing_date_column_raises(self):
        df = pd.DataFrame({"ISPU": [50.0, 60.0], "PM25": [30.0, 40.0]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            df.to_csv(f, index=False)
            tmp_path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="Kolom tanggal"):
                load_and_validate(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_missing_target_column_raises(self):
        df = pd.DataFrame({"Tanggal": ["2022-01-01"], "PM25": [30.0]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            df.to_csv(f, index=False)
            tmp_path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="Kolom target"):
                load_and_validate(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Smoke test: pipeline end-to-end
# ---------------------------------------------------------------------------

class TestPipelineSmoke:
    def test_end_to_end_small(self):
        """Pipeline harus berjalan tanpa error pada dataset kecil."""
        import argparse
        from pipeline import run_pipeline

        df_demo = generate_synthetic_ispu(n_days=120, seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = Path(tmpdir) / "test_input.csv"
            output_dir = Path(tmpdir) / "output"
            df_demo.to_csv(input_csv, index=False)

            args = argparse.Namespace(
                input=str(input_csv),
                output=str(output_dir),
                lag=1,
                train_frac=0.70,
                val_frac=0.15,
                anfis_features=2,  # kecil untuk kecepatan test
                n_mf=2,
                pso_particles=5,
                pso_iters=5,
                pso_patience=3,
                seed=42,
                verbose=False,
            )
            run_pipeline(args)

            # Verifikasi file output ada
            assert (output_dir / "predictions.csv").exists()
            assert (output_dir / "metrics.json").exists()
            assert (output_dir / "feature_importance.csv").exists()
            assert (output_dir / "pso_convergence.csv").exists()
            assert (output_dir / "hasil_prediksi_dashboard.csv").exists()

            # Verifikasi isi metrics.json valid
            import json
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            assert "random_forest" in metrics
            assert "anfis_pso" in metrics
            assert metrics["random_forest"]["MAE"] >= 0
            assert metrics["anfis_pso"]["MAE"] >= 0

            # Verifikasi predictions.csv memiliki data
            preds = pd.read_csv(output_dir / "predictions.csv")
            assert len(preds) > 0
            assert "ISPU_Aktual" in preds.columns
            assert "ISPU_Prediksi_RF" in preds.columns
            assert "ISPU_Prediksi_ANFIS_PSO" in preds.columns


if __name__ == "__main__":
    import sys
    # Jalankan tanpa pytest jika diperlukan
    print("Menjalankan smoke test pipeline...")
    test_instance = TestPipelineSmoke()
    test_instance.test_end_to_end_small()
    print("✓ Smoke test pipeline berhasil!")
    print("\nUntuk menjalankan semua unit test:")
    print("  pip install pytest")
    print("  pytest tests/test_pipeline.py -v")
