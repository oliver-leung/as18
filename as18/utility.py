import numpy as np


def lin_ind(basis: np.ndarray) -> bool:
    """Returns whether the input basis is linearly independent. (Unimplemented)"""
    ...


def gram_schmidt(B: np.ndarray) -> np.ndarray:
    """Returns the Gram-Schmidt orthogonalization of the input basis."""
    B_t = np.zeros(B.shape)
    for i in range(B.shape[0]):
        B_t[i] = np.copy(B[i])
        for j in range(i):
            mu_ji = np.dot(B[i], B_t[j]) / np.linalg.norm(B_t[j])**2
            B_t[i] -= mu_ji * B_t[j]
    return B_t
