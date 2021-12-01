from typing import List

import numpy as np
from matplotlib import pyplot as plt

from lattice import LatticePoint, Lattice, shortest, visualize, bin_by_coset
from utility import disjoint_pair


def partial_iter(points: List[LatticePoint], start: int, end: int, mod: int) -> List[LatticePoint]:
    """Perform one iteration of the averaging algorithm."""
    cp_to_vec = bin_by_coset(points, start, end, mod)

    # Naively select pairs within each coset
    pairs = []
    for vecs in cp_to_vec.values():
        pairs += disjoint_pair(vecs)

    # Get the sum & difference of each pair
    new_points = []
    for p1, p2 in pairs:
        new_points += [p1 + p2, p1 - p2]

    return new_points


def partial(lattice: Lattice) -> LatticePoint:
    iters = 4
    ldim = lattice.dim
    if ldim % iters != 0:
        raise ValueError('Currently only works with dimensions divisible by 4.')

    plt.ion()
    k = (ldim * 3) / 4  # Number of coords in the sliding window
    N = int(10 * iters * 2 ** k)  # Number of starting vectors

    iters = (np.log(ldim) + 2 * np.log(10)) / (2 * k / ldim - 1)
    iters = int(iters)
    iters = iters - (iters % 4)

    points = [lattice.sample_dgd() for _ in range(N)]
    for i in range(iters):
        # visualize(points)
        start = int((i * k) % ldim)
        end = int((start + k) % ldim)
        # print('Binning', start, 'to', end)
        new_points = partial_iter(points, start, end, 2**(i + 1))

        new_points_norms = [pt.norm ** 2 for pt in new_points]
        new_points_norms_means = np.mean(new_points_norms)
        print('Average of square norms:', new_points_norms_means)

        # Stop iteration if we've zeroed out all of the vectors
        if new_points_norms_means == 0:
            break

        points = new_points

    points = [pt / 8 ** (iters / 4) for pt in points]
    return shortest(points)
