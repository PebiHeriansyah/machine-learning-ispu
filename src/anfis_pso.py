"""
anfis_pso.py — Implementasi ANFIS Sugeno Orde-1 dan PSO untuk optimasi parameter premis.

ANFIS (Adaptive Neuro-Fuzzy Inference System):
  - Lapisan 1: Hitung derajat keanggotaan Gaussian per fitur per MF.
  - Lapisan 2: Hitung kekuatan aturan (T-norm, produk).
  - Lapisan 3: Normalisasi kekuatan aturan.
  - Lapisan 4: Konsekuen Sugeno orde-1: w_i * (a_i0 + a_i1*x1 + ... + a_in*xn).
  - Lapisan 5: Output akhir = penjumlahan.

Tiga fitur + dua MF per fitur = 2^3 = 8 aturan fuzzy (dikonfigurasi lewat argumen).

PSO mengoptimasi parameter premis (pusat c dan lebar sigma MF Gaussian).
Parameter konsekuen (koefisien polinomial Sugeno) dihitung dengan Least Squares
pada training data untuk setiap partikel/solusi premis.

Modul ini tidak bergantung pada TensorFlow/PyTorch. Hanya menggunakan numpy.

CATATAN AKADEMIK:
  - Implementasi ini untuk demonstrasi konsep ANFIS dan PSO.
  - Performa pada data sintetis tidak mewakili performa pada data ISPU nyata Bengkulu.
  - Untuk penelitian skripsi, gunakan data ISPU resmi yang tervalidasi.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian Membership Function
# ---------------------------------------------------------------------------

def gaussian_mf(x: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """Hitung derajat keanggotaan Gaussian: exp(-0.5 * ((x - c) / sigma)^2).

    Args:
        x: Array nilai input, shape (n_samples,).
        center: Pusat Gaussian (c).
        sigma: Lebar Gaussian; dikliping ke minimum 1e-6 agar tidak NaN.

    Returns:
        Array derajat keanggotaan, shape (n_samples,), nilai di [0, 1].
    """
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


# ---------------------------------------------------------------------------
# ANFIS Sugeno Orde-1
# ---------------------------------------------------------------------------

class ANFISSugeno:
    """ANFIS Sugeno orde-1 dengan MF Gaussian dan aturan produk.

    Args:
        n_features: Jumlah fitur masukan.
        n_mf: Jumlah MF per fitur (default 2).
        random_state: Seed untuk inisialisasi parameter premis.
    """

    def __init__(self, n_features: int, n_mf: int = 2, random_state: int = 42) -> None:
        if n_features < 1:
            raise ValueError("n_features harus >= 1")
        if n_mf < 1:
            raise ValueError("n_mf harus >= 1")
        self.n_features = n_features
        self.n_mf = n_mf
        self.n_rules = n_mf ** n_features
        self.random_state = random_state

        # Parameter premis: shape (n_features, n_mf, 2) — [center, sigma]
        rng = np.random.default_rng(random_state)
        centers = rng.uniform(0.0, 1.0, (n_features, n_mf))
        sigmas = np.full((n_features, n_mf), 0.5)
        self.premise_params: np.ndarray = np.stack([centers, sigmas], axis=-1)  # (F, M, 2)

        # Parameter konsekuen: (n_rules, n_features + 1) — [a0, a1, ..., an]
        # Diinisialisasi nol; diperbarui oleh fit_consequent.
        self.consequent_params: np.ndarray = np.zeros((self.n_rules, n_features + 1))

        # Indeks kombinasi aturan: (n_rules, n_features), setiap baris = [mf_idx fitur 0, mf_idx fitur 1, ...]
        self._rule_indices: np.ndarray = self._build_rule_indices()

        self._fitted = False

    def _build_rule_indices(self) -> np.ndarray:
        """Bangun indeks MF untuk setiap aturan dalam format lexicographic."""
        indices = np.array(
            np.meshgrid(*[np.arange(self.n_mf)] * self.n_features, indexing="ij")
        ).T.reshape(-1, self.n_features)
        return indices  # (n_rules, n_features)

    def _compute_rule_strengths(self, X: np.ndarray) -> np.ndarray:
        """Hitung kekuatan aturan ternormalisasi.

        Args:
            X: Array fitur, shape (n_samples, n_features).

        Returns:
            w_norm: Kekuatan ternormalisasi, shape (n_samples, n_rules).
        """
        n_samples = X.shape[0]
        # Hitung derajat keanggotaan semua MF: shape (n_samples, n_features, n_mf)
        mu = np.zeros((n_samples, self.n_features, self.n_mf), dtype=np.float64)
        for f in range(self.n_features):
            for m in range(self.n_mf):
                c = self.premise_params[f, m, 0]
                s = self.premise_params[f, m, 1]
                mu[:, f, m] = gaussian_mf(X[:, f], c, s)

        # Hitung kekuatan aturan sebagai produk MF per aturan: shape (n_samples, n_rules)
        w = np.ones((n_samples, self.n_rules), dtype=np.float64)
        for r, rule in enumerate(self._rule_indices):
            for f, mf_idx in enumerate(rule):
                w[:, r] *= mu[:, f, mf_idx]

        # Normalisasi; jika total nol pada sampel tertentu, bagi rata.
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum = np.where(w_sum < 1e-12, 1.0, w_sum)
        return w / w_sum  # (n_samples, n_rules)

    def _build_phi(self, X: np.ndarray, w_norm: np.ndarray) -> np.ndarray:
        """Bangun matriks regresi Phi untuk estimasi least-squares konsekuen.

        Phi[n, r*(n_features+1) + 0] = w_norm[n, r]         (a_r0)
        Phi[n, r*(n_features+1) + 1+j] = w_norm[n, r]*X[n,j]  (a_rj)

        Args:
            X: (n_samples, n_features)
            w_norm: (n_samples, n_rules)

        Returns:
            Phi: (n_samples, n_rules * (n_features + 1))
        """
        n_samples = X.shape[0]
        n_cols = self.n_rules * (self.n_features + 1)
        phi = np.zeros((n_samples, n_cols), dtype=np.float64)
        for r in range(self.n_rules):
            base = r * (self.n_features + 1)
            phi[:, base] = w_norm[:, r]
            for j in range(self.n_features):
                phi[:, base + 1 + j] = w_norm[:, r] * X[:, j]
        return phi

    def fit_consequent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Estimasi parameter konsekuen dengan Least Squares (pseudoinverse).

        Args:
            X: Fitur training, shape (n_samples, n_features).
            y: Target, shape (n_samples,).
        """
        if X.shape[0] < 2:
            logger.warning("Terlalu sedikit sampel untuk fit konsekuen; parameter konsekuen diset nol.")
            return
        w_norm = self._compute_rule_strengths(X)
        phi = self._build_phi(X, w_norm)
        # Least squares: theta = pinv(Phi) @ y
        theta, _, _, _ = np.linalg.lstsq(phi, y, rcond=None)
        # Reshape: (n_rules, n_features+1)
        self.consequent_params = theta.reshape(self.n_rules, self.n_features + 1)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Inferens: hitung output ANFIS untuk input X.

        Args:
            X: Fitur, shape (n_samples, n_features).

        Returns:
            y_pred: Prediksi, shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        w_norm = self._compute_rule_strengths(X)  # (n_samples, n_rules)

        # Output = sum_r w_norm_r * (a_r0 + sum_j a_rj * x_j)
        y_pred = np.zeros(X.shape[0], dtype=np.float64)
        for r in range(self.n_rules):
            # Konsekuen lokal: (n_samples,)
            f_r = self.consequent_params[r, 0]  # bias
            for j in range(self.n_features):
                f_r = f_r + self.consequent_params[r, 1 + j] * X[:, j]
            y_pred += w_norm[:, r] * f_r
        return y_pred

    def get_premise_flat(self) -> np.ndarray:
        """Kembalikan parameter premis sebagai vektor 1D untuk PSO.

        Returns:
            Vektor 1D, panjang n_features * n_mf * 2.
        """
        return self.premise_params.flatten().copy()

    def set_premise_flat(self, flat: np.ndarray) -> None:
        """Set parameter premis dari vektor 1D.

        Args:
            flat: Vektor 1D, panjang n_features * n_mf * 2.
        """
        expected = self.n_features * self.n_mf * 2
        if len(flat) != expected:
            raise ValueError(f"Panjang vektor premis harus {expected}, bukan {len(flat)}")
        self.premise_params = flat.reshape(self.n_features, self.n_mf, 2).copy()


# ---------------------------------------------------------------------------
# PSO Optimizer
# ---------------------------------------------------------------------------

class PSOOptimizer:
    """Particle Swarm Optimization untuk mengoptimasi parameter premis ANFIS.

    Fitness: RMSE pada validation set (lebih kecil = lebih baik).
    Boundary enforcement: clipping ke [lb, ub] setelah pembaruan posisi.

    Args:
        anfis: Objek ANFISSugeno yang akan dioptimasi.
        X_train: Fitur training untuk fit konsekuen.
        y_train: Target training.
        X_val: Fitur validasi untuk evaluasi fitness.
        y_val: Target validasi.
        n_particles: Jumlah partikel.
        max_iter: Maksimum iterasi.
        w: Inertia weight.
        c1: Cognitive coefficient.
        c2: Social coefficient.
        lb: Batas bawah parameter premis.
        ub: Batas atas parameter premis.
        random_state: Seed RNG.
        patience: Berhenti awal jika tidak ada perbaikan selama `patience` iterasi.
        tol: Toleransi perbaikan minimum fitness.
    """

    def __init__(
        self,
        anfis: ANFISSugeno,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_particles: int = 20,
        max_iter: int = 50,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        lb: float = -3.0,
        ub: float = 4.0,
        random_state: int = 42,
        patience: int = 10,
        tol: float = 1e-6,
    ) -> None:
        self.anfis = anfis
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.float64)
        self.X_val = np.asarray(X_val, dtype=np.float64)
        self.y_val = np.asarray(y_val, dtype=np.float64)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.lb = lb
        self.ub = ub
        self.random_state = random_state
        self.patience = patience
        self.tol = tol

        self.dim = anfis.n_features * anfis.n_mf * 2  # panjang vektor premis
        self.convergence_history: list[float] = []
        self.best_fitness: float = float("inf")
        self.best_params: Optional[np.ndarray] = None

    def _fitness(self, position: np.ndarray) -> float:
        """Evaluasi fitness: RMSE pada validation set.

        Parameter konsekuen di-fit baru untuk setiap posisi partikel
        menggunakan Least Squares pada training data.

        Args:
            position: Vektor parameter premis, shape (dim,).

        Returns:
            RMSE (float); float('inf') jika gagal.
        """
        try:
            self.anfis.set_premise_flat(position)
            self.anfis.fit_consequent(self.X_train, self.y_train)
            y_pred = self.anfis.predict(self.X_val)
            if not np.all(np.isfinite(y_pred)):
                return float("inf")
            rmse = float(np.sqrt(np.mean((self.y_val - y_pred) ** 2)))
            return rmse
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Fitness evaluation failed: %s", exc)
            return float("inf")

    def optimize(self) -> np.ndarray:
        """Jalankan PSO dan kembalikan parameter premis terbaik.

        Returns:
            Vektor parameter premis terbaik, shape (dim,).
        """
        rng = np.random.default_rng(self.random_state)

        # Inisialisasi posisi dan kecepatan
        positions = rng.uniform(self.lb, self.ub, (self.n_particles, self.dim))
        velocities = rng.uniform(
            -(self.ub - self.lb) * 0.1,
            (self.ub - self.lb) * 0.1,
            (self.n_particles, self.dim),
        )

        # Personal best
        p_best_pos = positions.copy()
        p_best_fit = np.array([self._fitness(pos) for pos in positions])

        # Global best
        g_best_idx = int(np.argmin(p_best_fit))
        g_best_pos = p_best_pos[g_best_idx].copy()
        g_best_fit = float(p_best_fit[g_best_idx])
        self.best_fitness = g_best_fit

        no_improve_count = 0
        for iteration in range(self.max_iter):
            r1 = rng.uniform(0.0, 1.0, (self.n_particles, self.dim))
            r2 = rng.uniform(0.0, 1.0, (self.n_particles, self.dim))

            # Pembaruan kecepatan (PSO standar)
            velocities = (
                self.w * velocities
                + self.c1 * r1 * (p_best_pos - positions)
                + self.c2 * r2 * (g_best_pos - positions)
            )

            # Clipping kecepatan untuk stabilitas
            v_max = (self.ub - self.lb) * 0.2
            velocities = np.clip(velocities, -v_max, v_max)

            # Pembaruan posisi + boundary enforcement
            positions = np.clip(positions + velocities, self.lb, self.ub)

            # Evaluasi fitness
            fits = np.array([self._fitness(pos) for pos in positions])

            # Update personal best
            improved = fits < p_best_fit
            p_best_fit = np.where(improved, fits, p_best_fit)
            p_best_pos = np.where(improved[:, None], positions, p_best_pos)

            # Update global best
            current_best_idx = int(np.argmin(p_best_fit))
            current_best_fit = float(p_best_fit[current_best_idx])
            self.convergence_history.append(current_best_fit)

            if current_best_fit < self.best_fitness - self.tol:
                self.best_fitness = current_best_fit
                g_best_pos = p_best_pos[current_best_idx].copy()
                g_best_fit = current_best_fit
                no_improve_count = 0
            else:
                no_improve_count += 1

            logger.info(
                "PSO iter %d/%d | best RMSE (val): %.4f",
                iteration + 1, self.max_iter, self.best_fitness,
            )

            if no_improve_count >= self.patience:
                logger.info(
                    "PSO berhenti awal pada iterasi %d (tidak ada perbaikan selama %d iterasi).",
                    iteration + 1, self.patience,
                )
                break

        self.best_params = g_best_pos.copy()
        return g_best_pos.copy()
