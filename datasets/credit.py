"""Credit dataset preparation for R-FairSC experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class CreditDataset:
    adjacency: sp.csr_matrix
    degree_inv_sqrt: sp.csr_matrix
    fairness_matrix: np.ndarray
    group_indicator: np.ndarray
    node_ids: np.ndarray
    group_values: np.ndarray
    group_counts: np.ndarray


def load_credit_dataset(
    edges_path: str | Path = "data/credit/credit_edges.csv",
    colors_path: str | Path = "data/credit/credit_colors.csv",
) -> CreditDataset:
    """Load the Credit graph, keep its largest component, and build F and G."""
    edges_path = Path(edges_path)
    colors_path = Path(colors_path)

    edges_df = pd.read_csv(edges_path, header=None, skiprows=1)
    edges = (
        edges_df.apply(pd.to_numeric, errors="coerce")
        .dropna()
        .astype(int)
        .to_numpy()
    )
    attributes = pd.read_csv(colors_path, header=None).to_numpy()

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

    protected_values = attributes[node_ids, 1]
    group_values = np.unique(protected_values)
    n = adjacency.shape[0]
    num_groups = len(group_values)

    group_indicator = np.zeros((n, num_groups), dtype=float)
    fairness_matrix = np.zeros((n, max(num_groups - 1, 0)), dtype=float)
    group_counts = np.zeros(num_groups, dtype=int)

    for group_index, group_value in enumerate(group_values):
        membership = protected_values == group_value
        count = int(membership.sum())
        group_counts[group_index] = count
        group_indicator[:, group_index] = membership.astype(float)

        if group_index < num_groups - 1:
            fairness_matrix[:, group_index] = membership.astype(float) - count / n

    return CreditDataset(
        adjacency=adjacency,
        degree_inv_sqrt=degree_inv_sqrt,
        fairness_matrix=fairness_matrix,
        group_indicator=group_indicator,
        node_ids=node_ids,
        group_values=group_values,
        group_counts=group_counts,
    )
