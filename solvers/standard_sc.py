"""Standard spectral clustering baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering as SC


@dataclass(frozen=True)
class SCSolution:
    labels: np.ndarray
    cluster_sizes: np.ndarray


class StandardSC:
    """SC with precomputed dense adjacency, matching the original baseline."""

    def __init__(self, adjacency: sp.spmatrix) -> None:
        self.adjacency = sp.csr_matrix(adjacency, dtype=np.float64)

    def fit_predict(self, k: int) -> SCSolution:
        labels = SC(
            n_clusters=int(k),
            affinity="precomputed",
        ).fit_predict(np.asarray(self.adjacency.todense()))

        return SCSolution(
            labels=labels,
            cluster_sizes=np.bincount(labels),
        )
