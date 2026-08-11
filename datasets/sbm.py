"""Synthetic stochastic block model used by the R-FairSC experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import networkx as nx
import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class SBMDataset:
    adjacency: sp.csr_matrix
    degree_inv_sqrt: sp.csr_matrix
    fairness_matrix: np.ndarray
    group_indicator: np.ndarray
    true_labels: np.ndarray


def _degree_inv_sqrt(adjacency: sp.spmatrix) -> sp.csr_matrix:
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
    positive = degrees > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    return sp.diags(inv_sqrt, offsets=0, shape=adjacency.shape, format="csr")


def group_indicator_from_fairness(fairness_matrix: np.ndarray) -> np.ndarray:
    """Reconstruct the h protected-group indicators from centered F."""
    fairness_matrix = np.asarray(fairness_matrix, dtype=float)
    n, h_minus_one = fairness_matrix.shape
    num_groups = h_minus_one + 1

    indicator = np.zeros((n, num_groups), dtype=int)
    assigned = np.zeros(n, dtype=bool)

    for group_index in range(h_minus_one):
        membership = fairness_matrix[:, group_index] > 0
        indicator[membership, group_index] = 1
        assigned |= membership

    indicator[~assigned, num_groups - 1] = 1
    return indicator


def generate_sbm_dataset(
    n: int,
    h: int,
    k: int,
    seed: int = 0,
) -> SBMDataset:
    """Generate the SBM exactly following the original experiment construction."""
    a_factor, b_factor, c_factor, d_factor = 20.0, 5.0, 5.0, 1.0
    scale = (log(n) / n) ** (2.0 / 3.0)
    a = a_factor * scale
    b = b_factor * scale
    c = c_factor * scale
    d = d_factor * scale

    base_block_size = n // (k * h)
    block_sizes = [base_block_size] * (k * h)
    for block_index in range(n - sum(block_sizes)):
        block_sizes[block_index] += 1

    probability_matrix = np.full((k * h, k * h), d, dtype=float)

    for cluster_i in range(k):
        for cluster_j in range(k):
            for group_i in range(h):
                for group_j in range(h):
                    row = cluster_i * h + group_i
                    col = cluster_j * h + group_j

                    if cluster_i == cluster_j:
                        probability_matrix[row, col] = a if group_i == group_j else c
                    elif group_i == group_j:
                        probability_matrix[row, col] = b

    graph = nx.stochastic_block_model(
        block_sizes,
        probability_matrix.tolist(),
        seed=seed,
    )

    true_labels = np.zeros(n, dtype=float)
    sensitive = np.zeros(n, dtype=float)

    # Preserve the original index-based label construction.
    for cluster_index in range(1, k + 1):
        for group_index in range(1, h + 1):
            start = int(
                ((n / k) * (cluster_index - 1))
                + ((n / (k * h)) * (group_index - 1))
            )
            end = int(
                ((n / k) * (cluster_index - 1))
                + ((n / (k * h)) * group_index)
            )
            sensitive[start:end] = group_index
            true_labels[start:end] = cluster_index

    sensitive_new = np.copy(sensitive)
    for new_value, old_value in enumerate(np.unique(sensitive), start=1):
        sensitive_new[sensitive == old_value] = new_value

    fairness_matrix = np.zeros((n, h - 1), dtype=float)
    for group_index in range(h - 1):
        membership = sensitive_new == group_index + 1
        group_size = np.sum(membership)
        fairness_matrix[:, group_index] = membership - group_size / n

    adjacency = nx.adjacency_matrix(graph).astype(np.float64).tocsr()
    degree_inv_sqrt = _degree_inv_sqrt(adjacency)
    group_indicator = group_indicator_from_fairness(fairness_matrix)

    return SBMDataset(
        adjacency=adjacency,
        degree_inv_sqrt=degree_inv_sqrt,
        fairness_matrix=fairness_matrix,
        group_indicator=group_indicator,
        true_labels=true_labels,
    )