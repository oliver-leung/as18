from functools import reduce

from numpy import gcd
from typing import List, Tuple

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


def partial_cycle(points: List[LatticePoint], window_size: int = 3, cycle_size: int = 4) -> Tuple[List[LatticePoint], LatticePoint]:
    dim = points[0].dim
    shortest_per_iter = []

    for i in range(cycle_size):
        # visualize(points)

        start = int((i * window_size) % dim)
        end = int((start + window_size) % dim)

        # print('Binning', start, 'to', end)
        new_points = partial_iter(points, start, end, 2 ** (i + 1))

        new_points_norms = [pt.norm ** 2 for pt in new_points]
        new_points_norms_means = np.mean(new_points_norms)
        print('Average of square norms:', new_points_norms_means)

        # Stop iteration if we've zeroed out all of the vectors
        if new_points_norms_means == 0:
            break

        shortest_per_iter.append(shortest(new_points))
        points = new_points

    coord_scale = int(2 ** (window_size / gcd(dim, window_size)))
    reduced_points = [pt / coord_scale for pt in points]
    # reduced_points = [pt / coord_scale for pt in points if gcd_many(pt.coords) % coord_scale == 0]
    shortest_per_iter.append(shortest(reduced_points))
    return reduced_points, shortest(shortest_per_iter)


def gcd_many(lst):
    return reduce(lambda x, y: gcd(x, y), lst)


def partial(lattice: Lattice, window_size: int = 3, s=10) -> LatticePoint:
    plt.ion()

    dim = lattice.dim
    cycle_size = int(dim / gcd(dim, window_size))

    iters = (np.log(dim) + 2 * np.log(10)) / (2 * window_size / dim - 1)  # The "l" parameter
    cycles = int(iters / cycle_size) + 3

    N = int(10 * iters * 2 ** window_size)  # Number of starting vectors
    points = [lattice.sample_dgd(s=s) for _ in range(N)]
    shortest_per_cycle = []

    for i in range(cycles):
        new_points, shortest_vec = partial_cycle(points, window_size=window_size, cycle_size=cycle_size)
        shortest_per_cycle.append(shortest_vec)

        new_points_norms = [pt.norm ** 2 for pt in new_points]
        new_points_norms_means = np.mean(new_points_norms)
        # print('Average of square norms:', new_points_norms_means)

        # Stop iteration if we've zeroed out all of the vectors
        if new_points_norms_means == 0:
            break

        points = new_points

    return shortest(shortest_per_cycle)
