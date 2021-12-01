from collections import defaultdict
from time import time
from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt

from lattice import Lattice, LatticePoint, QaryLattice
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


def partial_bin(points: List[LatticePoint], start: int, end: int, mod: int) -> Dict[str, List[LatticePoint]]:
    cp_to_vec = defaultdict(list)  # Mapping from Coord Parities to vector lists

    # Bin according to mod 2L
    for point in points:
        if start < end:
            parity_coords = point.coords[start: end]
        else:
            parity_coords = point.coords.take(range(start, end + 4), mode='wrap')

        coords_str = (parity_coords % mod).astype(int).astype(str)
        parity = ''.join(coords_str)

        cp_to_vec[parity].append(point)

    return cp_to_vec


def partial_iter(points: List[LatticePoint], start: int, end: int, mod: int) -> List[LatticePoint]:
    """Perform one iteration of the averaging algorithm."""
    cp_to_vec = partial_bin(points, start, end, mod)

    # Naively select pairs within each coset
    pairs = []
    for _, val in cp_to_vec.items():
        while len(val) > 1:
            p1 = val.pop(0)
            p2 = val.pop(0)
            pairs.append((p1, p2))

    # Get the average of each pair
    new_points = []
    for p1, p2 in pairs:
        pts_sum = p1 + p2
        pts_diff = p1 - p2
        new_points += [pts_sum, pts_diff]

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
        visualize(points)
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
