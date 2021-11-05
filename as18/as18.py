from time import time
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from lattice import Lattice, RealLattice, LatticePoint, IntegerLattice, QaryLattice


def avg(p1: LatticePoint, p2: LatticePoint) -> LatticePoint:
    """Take the average of two lattice points, provided that they are in the
    same coset.
    """
    assert np.array_equal(p1.basis, p2.basis)  # Naive lattice equality check

    coords = np.mean([p1.coords, p2.coords], axis=0)
    # TODO: Check if coords are still integers.
    point = LatticePoint(basis=p1.basis, coords=coords)

    return point


def diff_avg(p1: LatticePoint, p2: LatticePoint) -> LatticePoint:
    """Take the "difference average" of two lattice points, provided that they
    are in the same coset.
    """
    assert np.array_equal(p1.basis, p2.basis)  # Naive lattice equality check

    coords = (p1.coords - p2.coords) / 2
    # TODO: Check if coords are still integers.
    point = LatticePoint(basis=p1.basis, coords=coords)

    return point


def shortest(points: List[LatticePoint]) -> LatticePoint:
    """Find the shortest point within a list of points with the same dimensionality."""
    dim = points[0].dim
    zeros = np.zeros(dim)

    nonzero_pts = [pt for pt in points if not np.array_equal(pt.vec, zeros)]
    nonzero_pts_norms = [pt.norm for pt in nonzero_pts]
    shortest_pt_arg = np.argmin(nonzero_pts_norms)
    shortest_pt = nonzero_pts[shortest_pt_arg]

    return shortest_pt


def as18_iter(points: List[LatticePoint]) -> List[LatticePoint]:
    """Perform one iteration of the averaging algorithm."""
    cp_to_vec = {}  # Mapping from Coord Parities to vector lists

    # Bin according to mod 2L
    for point in points:
        coords_str = (point.coords % 2).astype(str)
        parity = ''.join(coords_str)

        if parity in cp_to_vec:
            cp_to_vec[parity].append(point)
        else:
            cp_to_vec[parity] = [point]

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
        pts_avg = avg(p1, p2)
        pts_diff_avg = diff_avg(p1, p2)
        new_points += [pts_avg, pts_diff_avg]

    return new_points


def visualize(lattice_pts: List[LatticePoint], noisy=True) -> None:
    if lattice_pts[0].dim > 3:
        raise ValueError('You can only plot lattices in R^3 or of lower dimensionality.')
    vectors = np.array([pt.vec for pt in lattice_pts]).T
    if noisy:
        vectors = np.array([vec + np.random.normal(0, 0.1, len(lattice_pts)) for vec in vectors])

    fig = plt.figure(num=0)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*vectors)
    plt.pause(0.5)


def as18(lattice: Lattice, N=1000, max_iters=50) -> LatticePoint:
    # Prepare MPL
    plt.ion()

    points = [lattice.sample_dgd() for _ in range(N)]
    # print('Points:\n', [point for point in points[:10]])

    # Perform iterations
    for _ in range(max_iters):
        if lattice.dim <= 3:
            visualize(points)
        new_points = as18_iter(points)

        # Stop iteration if we ran out of points
        if not new_points:
            print('Warning: did not start with enough points')
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
    start = time()
    basis = np.array([[1, 5, 4], [4, 5, -3], [7, 10, -9]])
    # lat = RealLattice(basis)
    lat = QaryLattice(q=100, dim=3)
    shortest_point = as18(lattice=lat, N=1000, max_iters=1000)
    print('Shortest point:', shortest_point)
    print('Took', time() - start, 'seconds')
