from typing import Union, List, Tuple

import numpy as np


def gaussian(s: float, x: Union[float, np.ndarray]) -> float:
    """Compute rho_s(x)."""
    norm = np.linalg.norm(x)
    rho = np.exp(-np.pi * norm ** 2 / s ** 2)
    return rho


def gram_schmidt(B: np.ndarray) -> np.ndarray:
    """Returns the Gram-Schmidt orthogonalization of the input basis."""
    B_t = np.zeros(B.shape)

    for i in range(B.shape[0]):
        B_t[i] = np.copy(B[i])

        # Remove from B_i the components of all of the preceding basis vectors
        for j in range(i):
            mu_ji = np.dot(B[i], B_t[j]) / np.linalg.norm(B_t[j]) ** 2
            B_t[i] -= mu_ji * B_t[j]

    return B_t


def validate_basis(vecs: np.ndarray):
    """Ensure that [vecs] is a linearly-independent, spanning basis."""
    if vecs.shape[0] != vecs.shape[1] or vecs.ndim != 2:
        raise ValueError(f'This basis is not square, or is not a 2-dimensional matrix: \n{vecs}\n')

    _, Sigma, _ = np.linalg.svd(vecs)
    if 0 in Sigma:
        raise ValueError(f'This basis is linearly dependent: \n{vecs}\n')

    if np.linalg.det(vecs) == 0:
        raise ValueError(f'This basis is singular: \n{vecs}\n')


def disjoint_pair(lst: List) -> List[Tuple]:
    return list(zip(lst[::2], lst[1::2]))