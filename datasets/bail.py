"""Bail dataset preparation for R-FairSC experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class BailDataset:
    adjacency: sp.csr_matrix
    degree_inv_sqrt: sp.csr_matrix
    fairness_matrix: np.ndarray
    group_indicator: np.ndarray
    node_ids: np.ndarray
    group_values: np.ndarray
    group_counts: np.ndarray


def load_bail_dataset(
    data_path: str | Path = "data/bail/bail.csv",
    edges_path: str | Path = "data/bail/bail_edges.txt",
    sensitive_attribute: str = "WHITE",
) -> BailDataset:
    """Load Bail, keep its largest component, and build F and G."""
    data_path = Path(data_path)
    edges_path = Path(edges_path)

    frame = pd.read_csv(data_path)
    protected_values_all = frame[sensitive_attribute].to_numpy().astype(int)
    edges = np.loadtxt(edges_path, dtype=int)
    edges = np.atleast_2d(edges)

    graph = nx.Graph()
    if edges.size:
        max_node = int(edges.max())
        graph.add_nodes_from(range(max_node + 1))
        graph.add_edges_from((int(i), int(j)) for i, j in edges)

    largest_component = max(nx.connected_components(graph), key=len)
    node_ids = np.array(sorted(largest_component), dtype=int)
    subgraph = graph.subgraph(node_ids)

    adjacency = nx.to_scipy_sparse_array(
        subgraph,
        nodelist=node_ids.tolist(),
        dtype=np.float64,
        format="csr",
    )
    adjacency = sp.csr_matrix(adjacency)

    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    degree_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees), format="csr")

    protected_values = protected_values_all[node_ids]
    group_values = np.sort(np.unique(protected_values))
    n = adjacency.shape[0]
    num_groups = len(group_values)

    group_indicator = np.zeros((n, num_groups), dtype=float)
    group_counts = np.zeros(num_groups, dtype=int)

    for group_index, group_value in enumerate(group_values):
        membership = protected_values == group_value
        group_indicator[:, group_index] = membership.astype(float)
        group_counts[group_index] = int(membership.sum())

    # Match the original Bail construction:
    # F = G[:, :h-1] - mean(G[:, :h-1], axis=0).
    fairness_matrix = group_indicator[:, : max(num_groups - 1, 0)].copy()
    if fairness_matrix.shape[1] > 0:
        fairness_matrix -= fairness_matrix.mean(axis=0, keepdims=True)

    return BailDataset(
        adjacency=adjacency,
        degree_inv_sqrt=degree_inv_sqrt,
        fairness_matrix=fairness_matrix,
        group_indicator=group_indicator,
        node_ids=node_ids,
        group_values=group_values,
        group_counts=group_counts,
    )