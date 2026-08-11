"""A-FairSC solver for fair spectral clustering.

"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class AFairSCConfig:
    """Hyperparameters for A-FairSC."""

    alpha0: float = 5e-3
    max_admm_iterations: int = 50
    tau: float = 2.0
    mu: float = 10.0
    lbfgs_max_iterations: int = 200
    lbfgs_gtol: float = 1e-3
    lbfgs_ftol: float = 1e-4
    warm_start_dual: bool = True
    eps_eig: float = 1e-12
    alpha_max: float = 0.99
    alpha_min: float = 1e-8


@dataclass
class AFairSCSolution:
    """Output of one A-FairSC run."""

    labels: np.ndarray
    embedding: np.ndarray
    history: list[dict]
    h_runtime: float
    fairness_violation: float
    cluster_sizes: np.ndarray


class AFairSC:
    """A-FairSC solver for fair spectral clustering."""

    def __init__(
        self,
        adjacency: sp.spmatrix,
        degree_inv_sqrt: sp.spmatrix,
        fairness_matrix: np.ndarray | sp.spmatrix,
        config: AFairSCConfig | None = None,
        verbose: int = 0,
    ) -> None:
        self.adjacency = sp.csr_matrix(adjacency)
        self.degree_inv_sqrt = sp.csr_matrix(degree_inv_sqrt)
        self.fairness_matrix = fairness_matrix
        self.config = config or AFairSCConfig()
        self.verbose = verbose

        self.n = self.adjacency.shape[0]
        self.normalized_affinity = (
            self.degree_inv_sqrt @ self.adjacency @ self.degree_inv_sqrt
        ).tocsr()
        self._null_projector_basis = self._build_null_projector_basis()

    def _build_null_projector_basis(self) -> np.ndarray:
        """Build an orthonormal basis for col(D^{-1/2} F)."""
        if sp.issparse(self.fairness_matrix):
            fairness_scaled = (
                self.degree_inv_sqrt @ self.fairness_matrix
            ).toarray()
        else:
            fairness_scaled = self.degree_inv_sqrt @ np.asarray(
                self.fairness_matrix,
                dtype=float,
            )

        q, r, _ = scipy.linalg.qr(
            fairness_scaled,
            mode="economic",
            pivoting=True,
        )
        diagonal = np.abs(np.diag(r)) if r.size else np.array([])
        if diagonal.size == 0:
            return np.zeros((self.n, 0), dtype=float)

        threshold = 1e-12 * diagonal.max()
        rank = int(np.sum(diagonal > threshold))
        return q[:, :rank]

    def _project_null_ft(self, matrix: np.ndarray) -> np.ndarray:
        basis = self._null_projector_basis
        if basis.shape[1] == 0:
            return matrix
        return matrix - basis @ (basis.T @ matrix)

    def _g_star_and_grad(self, v: np.ndarray) -> tuple[float, np.ndarray]:
        m_v = self.normalized_affinity @ v
        m2_v = self.normalized_affinity @ m_v

        gram = m_v.T @ m_v
        gram = 0.5 * (gram + gram.T)

        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        eigenvalues = np.maximum(eigenvalues, self.config.eps_eig)
        sqrt_eigenvalues = np.sqrt(eigenvalues)

        value = float(np.sum(sqrt_eigenvalues))
        inv_sqrt = 1.0 / sqrt_eigenvalues
        gram_inv_sqrt = (eigenvectors * inv_sqrt) @ eigenvectors.T
        gradient = m2_v @ gram_inv_sqrt
        return value, gradient

    def _phi_star_and_grad(
        self,
        v: np.ndarray,
        y: np.ndarray,
        dual: np.ndarray,
        alpha: float,
    ) -> tuple[float, np.ndarray]:
        alpha = float(min(alpha, self.config.alpha_max))
        eta = 1.0 / (1.0 - alpha)
        a = eta * (v + dual - alpha * y)

        term1 = 0.5 * np.sum(v * v)
        term2 = -0.5 * np.sum((a - v) * (a - v))
        term3 = float(np.sum(dual * a))
        term4 = 0.5 * alpha * np.sum((a - y) * (a - y))

        value = float(term1 + term2 + term3 + term4)
        gradient = a
        return value, gradient

    def _solve_dual_lbfgs(
        self,
        y: np.ndarray,
        dual: np.ndarray,
        alpha: float,
        k: int,
        seed: int,
        v_init: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        rng = np.random.default_rng(seed)
        if v_init is None:
            v0 = rng.standard_normal(size=(self.n, k))
        else:
            v0 = v_init.copy()

        alpha = float(
            np.clip(alpha, self.config.alpha_min, self.config.alpha_max)
        )

        def objective_and_gradient(v_flat: np.ndarray):
            v = v_flat.reshape(self.n, k)
            phi, grad_phi = self._phi_star_and_grad(v, y, dual, alpha)
            g_value, grad_g = self._g_star_and_grad(v)
            objective = phi - g_value
            gradient = grad_phi - grad_g
            return objective, gradient.reshape(-1)

        result = minimize(
            fun=objective_and_gradient,
            x0=v0.reshape(-1),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.config.lbfgs_max_iterations,
                "gtol": self.config.lbfgs_gtol,
                "ftol": self.config.lbfgs_ftol,
            },
        )

        v_hat = result.x.reshape(self.n, k)
        info = {
            "success": bool(result.success),
            "niter": int(result.nit),
            "final_obj": float(result.fun),
        }
        return v_hat, info

    def _recover_h_from_v(self, v_hat: np.ndarray) -> np.ndarray:
        transformed = self.normalized_affinity @ v_hat
        u, _, vt = np.linalg.svd(transformed, full_matrices=False)
        return u @ vt

    def fit_predict(self, k: int, seed: int = 0) -> AFairSCSolution:
        """Run A-FairSC for one cluster count and random seed."""
        k = int(k)
        config = self.config

        h = np.zeros((self.n, k), dtype=float)
        y = np.zeros((self.n, k), dtype=float)
        dual = np.zeros((self.n, k), dtype=float)
        alpha = float(np.clip(config.alpha0, config.alpha_min, config.alpha_max))

        v_warm: np.ndarray | None = None
        history: list[dict] = []
        h_runtime = 0.0

        for iteration in range(config.max_admm_iterations):
            alpha = float(np.clip(alpha, config.alpha_min, config.alpha_max))

            h_start = time.perf_counter()
            v_hat, dual_info = self._solve_dual_lbfgs(
                y=y,
                dual=dual,
                alpha=alpha,
                k=k,
                seed=seed + iteration,
                v_init=(
                    v_warm
                    if config.warm_start_dual and v_warm is not None
                    else None
                ),
            )
            if config.warm_start_dual:
                v_warm = v_hat

            h = self._recover_h_from_v(v_hat)
            h_runtime += time.perf_counter() - h_start

            m_h = self.normalized_affinity @ h
            y_new = self._project_null_ft(m_h + (1.0 / alpha) * dual)
            dual_new = dual + alpha * (m_h - y_new)

            primal_residual = m_h - y
            dual_residual = alpha * (y - y_new)
            primal_norm = float(np.linalg.norm(primal_residual, ord="fro"))
            dual_norm = float(np.linalg.norm(dual_residual, ord="fro"))

            alpha_new = alpha
            if primal_norm > config.mu * dual_norm:
                alpha_new = config.tau * alpha
            elif dual_norm > config.mu * primal_norm:
                alpha_new = alpha / config.tau

            alpha_new = float(
                np.clip(alpha_new, config.alpha_min, config.alpha_max)
            )

            history.append(
                {
                    "iter": iteration,
                    "alpha": alpha,
                    "rnorm": primal_norm,
                    "snorm": dual_norm,
                    "lbfgs": dual_info,
                }
            )

            if self.verbose:
                print(
                    f"[A-FairSC {iteration:02d}] "
                    f"alpha={alpha:.3e} "
                    f"r={primal_norm:.3e} "
                    f"s={dual_norm:.3e} "
                    f"lbfgs_it={dual_info['niter']} "
                    f"ok={dual_info['success']}"
                )

            y, dual, alpha = y_new, dual_new, alpha_new

        embedding = np.asarray(self.degree_inv_sqrt @ h)
        fairness_array = (
            self.fairness_matrix.toarray()
            if sp.issparse(self.fairness_matrix)
            else np.asarray(self.fairness_matrix)
        )
        fairness_violation = float(
            np.linalg.norm(fairness_array.T @ embedding, ord="fro")
        )

        labels = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=seed,
        ).fit_predict(embedding)
        cluster_sizes = np.bincount(labels, minlength=k)

        return AFairSCSolution(
            labels=labels,
            embedding=embedding,
            history=history,
            h_runtime=h_runtime,
            fairness_violation=fairness_violation,
            cluster_sizes=cluster_sizes,
        )