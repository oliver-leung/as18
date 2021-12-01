from time import time
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from lattice import Lattice, LatticePoint, QaryLattice
from partial import partial
from utility import shortest, bin_by_parity, visualize


def as18_iter(points: List[LatticePoint]) -> List[LatticePoint]:
    """Perform one iteration of the averaging algorithm."""
    coord_parity_to_vecs = bin_by_parity(points)

    # Naively select pairs within each coset
    pairs = []
    for vecs in coord_parity_to_vecs.values():
        pairs += list(zip(vecs[::2], vecs[1::2]))

    # Get the (difference-) average of each pair
    new_points = []
    for p1, p2 in pairs:
        new_points += [
            (p1 + p2) / 2,  # Average
            (p1 - p2) / 2   # Difference-average
        ]

    return new_points


def as18(lattice: Lattice, N=1000, max_iters=50) -> LatticePoint:
    # Prepare MPL
    plt.ion()

    points = [lattice.sample_dgd() for _ in range(N)]
    # print('Points:\n', [point for point in points[:10]])

    # Perform iterations
    for _ in range(max_iters):
        # if lattice.dim <= 3:
        visualize(points)
        new_points = as18_iter(points)

        # Stop iteration if we ran out of points
        if not new_points:
            print('Warning: did not start_time with enough points')
            break

        new_points_norms = [pt.norm ** 2 for pt in new_points]
        new_points_norms_means = np.mean(new_points_norms)
        print('Average of square norms:', new_points_norms_means)
        # print('New Points:', [point for point in new_points])

        # Stop iteration if we've zeroed out all of the vectors
        if new_points_norms_means == 0:
            break

        points = new_points

    return shortest(points)


if __name__ == '__main__':
    start_time = time()
    basis = np.array([[1, 5, 4], [4, 5, -3], [7, 10, -9]])
    # lat = RealLattice(basis)
    lat = QaryLattice(q=10, dim=4)
    # lat = IntegerLattice(dim=4)
    shortest_point = as18(lattice=lat, N=1000, max_iters=1000)
    print('Shortest point from AS18:', shortest_point)
    shortest_point = partial(lat)
    print('Shortest point from partial binning:', shortest_point)
    print('Took', time() - start_time, 'seconds')
