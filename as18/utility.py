from typing import Union, List

import numpy as np
from matplotlib import pyplot as plt

from lattice import LatticePoint


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


def shortest(points: List[LatticePoint]) -> LatticePoint:
    """Find the shortest point within a list of points with the same dimensionality."""
    dim = points[0].dim
    zeros = np.zeros(dim)

    nonzero_pts = [pt for pt in points if not np.array_equal(pt.vec, zeros)]
    nonzero_pts_norms = [pt.norm for pt in nonzero_pts]
    shortest_pt_arg = np.argmin(nonzero_pts_norms)
    shortest_pt = nonzero_pts[shortest_pt_arg]

    return shortest_pt


def bin_by_parity(points):
    cp_to_vec = {}  # Mapping from Coord Parities to vector lists
    # Bin according to mod 2L
    for point in points:
        coords_str = (point.coords % 2).astype(str)
        parity = ''.join(coords_str)

        if parity in cp_to_vec:
            cp_to_vec[parity].append(point)
        else:
            cp_to_vec[parity] = [point]
    return cp_to_vec


def visualize(lattice_pts: List[LatticePoint], noisy=True) -> None:
    vectors = np.array([pt.vec for pt in lattice_pts]).T
    if noisy:
        vectors = np.array([vec + np.random.normal(0, 0.1, len(lattice_pts)) for vec in vectors])

    fig = plt.figure(num=0)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*(vectors[:3]))
    plt.pause(0.5)