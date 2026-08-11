"""Evaluation metrics for fair graph clustering experiments."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from itertools import permutations

def normalized_cut(adjacency: sp.spmatrix, cluster_labels: np.ndarray) -> float:
    """Compute the normalized-cut objective for a hard clustering."""
    cluster_labels = np.asarray(cluster_labels)
    n = cluster_labels.size
    k = int(cluster_labels.max()) + 1

    assignment = sp.coo_matrix(
        (np.ones(n, dtype=int), (np.arange(n), cluster_labels)),
        shape=(n, k),
    ).tocsr()

    cluster_degrees = adjacency.dot(assignment).toarray()
    degrees = cluster_degrees.sum(axis=1)
    volumes = np.array(
        [degrees[cluster_labels == cluster_id].sum() for cluster_id in range(k)]
    )
    associations = (cluster_degrees * assignment.toarray()).sum(axis=0)
    cuts = volumes - associations

    return float(np.sum(cuts / volumes))


def compute_balance(
    cluster_labels: np.ndarray,
    group_indicator: np.ndarray,
    k: int,
    verbose: bool = False,
) -> float:
    """Compute the mean min/max protected-group balance across clusters."""
    cluster_labels = np.asarray(cluster_labels)
    group_indicator = np.asarray(group_indicator)

    balances = np.zeros(k, dtype=float)

    for cluster_id in range(k):
        in_cluster = cluster_labels == cluster_id
        group_counts = group_indicator[in_cluster].sum(axis=0)

        if verbose:
            for group_id, count in enumerate(group_counts, start=1):
                print(
                    f"--group #: {group_id}, "
                    f"count in cluster # {cluster_id + 1}: {count:g}"
                )

        max_count = float(group_counts.max()) if group_counts.size else 0.0
        min_count = float(group_counts.min()) if group_counts.size else 0.0
        if max_count > 0:
            balances[cluster_id] = min_count / max_count

    return float(balances.mean())
def clustering_accuracy(
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
) -> float:
    """Return the minimum mismatch rate over cluster-label permutations.

    This preserves the original SBM helper name and behavior: despite the
    function name, the returned value is an error rate, so lower is better.
    """
    true_labels = np.asarray(true_labels).flatten()
    cluster_labels = np.asarray(cluster_labels).flatten()

    unique_true = np.unique(true_labels)
    normalized_true = np.searchsorted(unique_true, true_labels) + 1

    unique_pred = np.unique(cluster_labels)
    normalized_pred = np.searchsorted(unique_pred, cluster_labels) + 1

    num_true = len(unique_true)
    num_pred = len(unique_pred)
    num_labels = max(num_true, num_pred)
    best_error = np.inf

    for permutation in permutations(range(1, num_labels + 1), num_labels):
        permuted = np.array([permutation[index - 1] for index in normalized_pred])
        error = np.mean(permuted != normalized_true)
        if error < best_error:
            best_error = error

    return float(best_error)