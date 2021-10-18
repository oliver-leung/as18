from math import ceil, floor
from random import randint, random

import numpy as np


def gram_schmidt(B: np.ndarray) -> np.ndarray:
    """Returns the Gram-Schmidt orthogonalization of the input basis."""
    B_t = np.zeros(B.shape)

    for i in range(B.shape[0]):
        B_t[i] = np.copy(B[i])
        for j in range(i):
            mu_ji = np.dot(B[i], B_t[j]) / np.linalg.norm(B_t[j]) ** 2
            B_t[i] -= mu_ji * B_t[j]

    return B_t


def sample_dgd_Z(s: float = 10, t: float = 5, c: float = 0) -> int:
    """
    Sample from the discrete Gaussian over some integer range.

    Args:
        s (float, optional): The Gaussian parameter (i.e. standard deviation).
        t (float, optional): The scaling factor, which determines the range of sampling.
        c (float, optional): The center around which to sample the discrete Gaussian.
    """
    while True:
        max_int = ceil(c + t * s)
        min_int = floor(c - t * s)
        z = randint(min_int, max_int)

        if np.exp(-np.pi * (z - c) ** 2 / s ** 2) <= random():
            break

    return z


def validate_basis(basis: np.ndarray):
    if basis.shape[0] != basis.shape[1] or basis.ndim != 2:
        raise ValueError(f'This basis is not square, or is not a 2-dimensional matrix: \n{basis}\n')
    if 0 in np.linalg.eig(basis)[1]:
        raise ValueError(f'This basis is linearly dependent: \n{basis}\n')
    if np.linalg.det(basis) == 0:
        raise ValueError(f'This basis is singular: \n{basis}\n')
