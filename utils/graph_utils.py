"""Graph and sparse-matrix utilities used by R-FairSC."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def normalized_laplacian(
    adjacency: sp.spmatrix,
    degree_inv_sqrt: sp.spmatrix,
) -> sp.csr_matrix:
    """Return the symmetric normalized graph Laplacian."""
    n = adjacency.shape[0]
    laplacian = sp.eye(n, format="csr") - degree_inv_sqrt @ adjacency @ degree_inv_sqrt
    laplacian = 0.5 * (laplacian + laplacian.T)
    return laplacian.tocsr()


def orthonormal_basis_col_f(
    fairness_matrix,
    eps: float = 1e-12,
    dtype=np.float32,
) -> np.ndarray:
    """Return an orthonormal basis for the column space of F."""
    if sp.issparse(fairness_matrix):
        f_sparse = fairness_matrix.tocsr()
    else:
        f_sparse = sp.csr_matrix(np.asarray(fairness_matrix, dtype=np.float64))

    ftf = (f_sparse.T @ f_sparse).toarray()
    ftf = 0.5 * (ftf + ftf.T)

    eigenvalues, eigenvectors = np.linalg.eigh(ftf)
    if eigenvalues.size == 0:
        return np.zeros((f_sparse.shape[0], 0), dtype=dtype)

    max_eigenvalue = float(eigenvalues.max())
    if max_eigenvalue <= 0:
        return np.zeros((f_sparse.shape[0], 0), dtype=dtype)

    keep = eigenvalues > eps * max_eigenvalue
    if not np.any(keep):
        return np.zeros((f_sparse.shape[0], 0), dtype=dtype)

    kept_values = eigenvalues[keep]
    kept_vectors = eigenvectors[:, keep]
    whitening = kept_vectors @ np.diag(1.0 / np.sqrt(kept_values))
    basis = (f_sparse @ whitening).astype(dtype, copy=False)
    return np.asarray(basis)


def scipy_csr_to_torch_sparse(
    matrix: sp.spmatrix,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a coalesced PyTorch COO tensor."""
    matrix_coo = matrix.tocoo()
    indices = np.vstack((matrix_coo.row, matrix_coo.col))

    indices_t = torch.tensor(indices, dtype=torch.long, device=device)
    values_t = torch.tensor(matrix_coo.data, dtype=dtype, device=device)

    return torch.sparse_coo_tensor(
        indices_t,
        values_t,
        size=matrix_coo.shape,
        device=device,
        dtype=dtype,
    ).coalesce()
