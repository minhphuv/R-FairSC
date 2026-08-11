"""R-FairSC: Riemannian optimization for fair spectral clustering."""

from __future__ import annotations

from dataclasses import dataclass
import time

import geoopt
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
import torch

from utils.graph_utils import (
    normalized_laplacian,
    orthonormal_basis_col_f,
    scipy_csr_to_torch_sparse,
)


@dataclass(frozen=True)
class RFairSCConfig:
    """Optimization parameters for R-FairSC."""

    alpha0: float = 5e-3
    max_admm_iterations: int = 50
    tau: float = 2.0
    mu: float = 2.0

    max_rcg_iterations: int = 200
    min_step_size: float = 1e-5
    grad_tol: float = 1e-5

    line_search_c: float = 1e-4
    line_search_rho: float = 0.5
    line_search_max_steps: int = 25

    primal_tol: float = 1e-4
    fairness_tol: float = 1e-4

    kmeans_n_init: int = 10
    kmeans_random_state: int = 0


@dataclass(frozen=True)
class RFairSCResult:
    """Result returned by one R-FairSC run."""

    labels: np.ndarray
    cluster_sizes: np.ndarray
    h_runtime: float
    fairness_violation: float
    admm_iterations: int
    final_alpha: float


class RFairSC:
    """R-FairSC solver with graph-dependent quantities precomputed once."""

    def __init__(
        self,
        adjacency: sp.spmatrix,
        degree_inv_sqrt: sp.spmatrix,
        fairness_matrix,
        config: RFairSCConfig | None = None,
        device: str | None = None,
        verbose: int = 0,
    ) -> None:
        self.config = config or RFairSCConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.n = adjacency.shape[0]
        self.manifold = geoopt.manifolds.EuclideanStiefel()

        laplacian = normalized_laplacian(adjacency, degree_inv_sqrt)
        self.laplacian = scipy_csr_to_torch_sparse(
            laplacian,
            device=self.device,
            dtype=torch.float32,
        )

        basis_np = orthonormal_basis_col_f(
            fairness_matrix,
            eps=1e-12,
            dtype=np.float32,
        )
        self.fairness_basis = torch.tensor(
            basis_np,
            dtype=torch.float32,
            device=self.device,
        )
        self.fairness_rank = self.fairness_basis.shape[1]

    def _project_null_ft(self, matrix: torch.Tensor) -> torch.Tensor:
        """Project a matrix onto null(F^T) using an orthonormal basis of col(F)."""
        if self.fairness_rank == 0:
            return matrix
        u = self.fairness_basis
        return matrix - u @ (u.T @ matrix)

    def _sync_cuda(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(device=self.device)

    @torch.no_grad()
    def _riemannian_cg_stiefel(
        self,
        h0: torch.Tensor,
        y: torch.Tensor,
        dual: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Solve the H-subproblem by Fletcher-Reeves RCG on the Stiefel manifold."""
        cfg = self.config
        h = h0.clone()

        def cost_and_euclidean_gradient(current_h: torch.Tensor):
            lh = torch.sparse.mm(self.laplacian, current_h)
            laplacian_term = torch.trace(current_h.T @ lh)

            difference = current_h - y + dual
            penalty_term = 0.5 * alpha * torch.sum(difference * difference)
            cost = laplacian_term + penalty_term

            euclidean_gradient = 2.0 * lh + alpha * difference
            return cost, euclidean_gradient

        cost, euclidean_gradient = cost_and_euclidean_gradient(h)
        riemannian_gradient = self.manifold.egrad2rgrad(h, euclidean_gradient)
        gradient_norm = torch.linalg.norm(riemannian_gradient)
        direction = -riemannian_gradient

        for _ in range(cfg.max_rcg_iterations):
            if gradient_norm.item() < cfg.grad_tol:
                break

            directional_derivative = torch.sum(riemannian_gradient * direction)
            if directional_derivative.item() > 0:
                direction = -riemannian_gradient
                directional_derivative = -torch.sum(
                    riemannian_gradient * riemannian_gradient
                )

            step_size = 1.0
            old_cost = cost
            backtracking_steps = 0

            while True:
                candidate = self.manifold.retr(h, step_size * direction)
                candidate_cost, _ = cost_and_euclidean_gradient(candidate)

                if candidate_cost <= (
                    old_cost
                    + cfg.line_search_c * step_size * directional_derivative
                ):
                    break

                step_size *= cfg.line_search_rho
                backtracking_steps += 1

                if (
                    step_size < cfg.min_step_size
                    or backtracking_steps >= cfg.line_search_max_steps
                ):
                    break

            h_new = self.manifold.retr(h, step_size * direction)
            cost_new, euclidean_gradient_new = cost_and_euclidean_gradient(h_new)
            riemannian_gradient_new = self.manifold.egrad2rgrad(
                h_new,
                euclidean_gradient_new,
            )

            numerator = torch.sum(riemannian_gradient_new * riemannian_gradient_new)
            denominator = torch.sum(riemannian_gradient * riemannian_gradient) + 1e-16
            beta = (numerator / denominator).clamp(min=0.0)

            transported_direction = self.manifold.proju(h_new, direction)
            direction = -riemannian_gradient_new + beta * transported_direction

            h = h_new
            cost = cost_new
            riemannian_gradient = riemannian_gradient_new
            gradient_norm = torch.linalg.norm(riemannian_gradient)

        return h

    @torch.no_grad()
    def fit_predict(self, k: int) -> RFairSCResult:
        """Run R-FairSC for a requested number of clusters."""
        cfg = self.config

        h = self.manifold.random(
            (self.n, k),
            dtype=torch.float32,
            device=self.device,
        )
        y = self._project_null_ft(h.clone())
        dual = torch.zeros((self.n, k), dtype=torch.float32, device=self.device)

        alpha = float(cfg.alpha0)
        h_runtime = 0.0
        fairness_violation = float("inf")
        admm_iterations = 0

        for iteration in range(cfg.max_admm_iterations):
            y_previous = y

            # H-step: RCG on the Stiefel manifold.
            self._sync_cuda()
            start = time.perf_counter()
            h_new = self._riemannian_cg_stiefel(
                h0=h,
                y=y,
                dual=dual,
                alpha=alpha,
            )
            self._sync_cuda()
            h_runtime += time.perf_counter() - start

            # Y-step: projection onto null(F^T).
            y_new = self._project_null_ft(h_new + dual)

            # Scaled dual update.
            dual_new = dual + (h_new - y_new)

            primal_residual = h_new - y_new
            primal_norm = torch.linalg.norm(primal_residual).item()

            dual_residual = alpha * (y_new - y_previous)
            dual_norm = torch.linalg.norm(dual_residual).item()

            fairness_violation = (
                0.0
                if self.fairness_rank == 0
                else torch.linalg.norm(self.fairness_basis.T @ h_new).item()
            )

            if self.verbose > 0:
                lh = torch.sparse.mm(self.laplacian, h_new)
                objective = torch.trace(h_new.T @ lh).item()
                print(
                    f"[R-FairSC] iter={iteration:02d} "
                    f"obj={objective:.6e} "
                    f"||R||={primal_norm:.3e} "
                    f"||S||={dual_norm:.3e} "
                    f"||U^T H||={fairness_violation:.3e} "
                    f"alpha={alpha:.3e} "
                    f"rank(F)={self.fairness_rank}"
                )

            if primal_norm > cfg.mu * dual_norm:
                alpha *= cfg.tau
            elif dual_norm > cfg.mu * primal_norm:
                alpha /= cfg.tau

            h, y, dual = h_new, y_new, dual_new
            admm_iterations = iteration + 1

            if (
                primal_norm < cfg.primal_tol
                and fairness_violation < cfg.fairness_tol
            ):
                break

        embedding = h.detach().cpu().numpy()
        row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        normalized_embedding = embedding / row_norms

        labels = KMeans(
            n_clusters=k,
            n_init=cfg.kmeans_n_init,
            random_state=cfg.kmeans_random_state,
        ).fit_predict(normalized_embedding)

        cluster_sizes = np.bincount(labels, minlength=k)

        return RFairSCResult(
            labels=labels,
            cluster_sizes=cluster_sizes,
            h_runtime=h_runtime,
            fairness_violation=fairness_violation,
            admm_iterations=admm_iterations,
            final_alpha=alpha,
        )
