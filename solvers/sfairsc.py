"""Scalable Fair Spectral Clustering (sFairSC)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigs
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class SFairSCSolution:
    labels: np.ndarray
    embedding: np.ndarray
    cluster_sizes: np.ndarray


class SFairSC:
    """sFairSC using the same projected operator as the original code."""

    def __init__(
        self,
        adjacency: sp.spmatrix,
        degree_inv_sqrt: sp.spmatrix,
        fairness_matrix: np.ndarray,
        eig_max_iterations: int = 1000,
        kmeans_max_iterations: int = 500,
    ) -> None:
        self.adjacency = sp.csr_matrix(adjacency, dtype=np.float64)
        self.degree_inv_sqrt = sp.csr_matrix(degree_inv_sqrt, dtype=np.float64)
        self.fairness_matrix = np.asarray(fairness_matrix, dtype=np.float64)
        self.eig_max_iterations = int(eig_max_iterations)
        self.kmeans_max_iterations = int(kmeans_max_iterations)

        self.n = self.adjacency.shape[0]

        # Equivalent to the original:
        #   L = D - W
        #   sqrtD = sqrtm(D)
        #   C = solve(sqrtD, F)
        #   Ln = solve(sqrtD, L @ inv(sqrtD))
        # Since D is diagonal, D^{-1/2} is already available from the loader.
        identity = sp.eye(self.n, format="csr", dtype=np.float64)
        normalized_affinity = self.degree_inv_sqrt @ self.adjacency @ self.degree_inv_sqrt
        self.normalized_laplacian = (identity - normalized_affinity).tocsr()
        self.normalized_laplacian = 0.5 * (
            self.normalized_laplacian + self.normalized_laplacian.T
        )

        constraints = self.degree_inv_sqrt @ self.fairness_matrix
        self.constraints = np.asarray(constraints, dtype=np.float64)
        self.sigma = float(np.linalg.norm(self.normalized_laplacian.toarray(), 1))

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        # Original Afun implementation.
        y1 = np.linalg.lstsq(self.constraints, vector, rcond=None)[0]
        y2 = vector - self.constraints @ y1
        y3 = self.normalized_laplacian @ y2
        y4 = np.linalg.lstsq(self.constraints, y3, rcond=None)[0]
        return y3 - self.constraints @ y4 - self.sigma * y2 + self.sigma * vector

    def fit_predict(self, k: int) -> SFairSCSolution:
        k = int(k)

        operator = LinearOperator((self.n, self.n), matvec=self._matvec)
        _, eigenvectors = eigs(
            operator,
            k=k,
            which="SR",
            maxiter=self.eig_max_iterations,
            ncv=4 * k,
        )

        embedding = self.degree_inv_sqrt @ eigenvectors.real
        embedding = np.asarray(embedding, dtype=np.float64)

        labels = KMeans(
            n_clusters=k,
            n_init=10,
            max_iter=self.kmeans_max_iterations,
        ).fit_predict(embedding)

        return SFairSCSolution(
            labels=labels,
            embedding=embedding,
            cluster_sizes=np.bincount(labels),
        )
