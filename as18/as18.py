from lattice import RealLattice, LatticePoint, IntegerLattice
import numpy as np
from numpy import linalg as LA
from typing import List
from time import time


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
    """Find the shortest point within a list of points."""
    # print([point for point in points])
    dim = points[0].dim
    zeros = np.zeros(dim)

    nonzero_points = [
        point for point in points if not np.array_equal(point.vec, zeros)]
    nonzero_points_norms = [point.norm for point in nonzero_points]
    shortest_point_arg = np.argmin(nonzero_points_norms)
    shortest_pt = nonzero_points[shortest_point_arg]

    return shortest_pt


def as18_iter(points: List[LatticePoint]) -> List[LatticePoint]:
    """Perform one iteration of the averaging algorithm."""
    cp_to_vec = {}

    # Bin according to mod 2L
    for point in points:
        coords_str = (point.coords % 2).astype(str)
        parity = ''.join(coords_str)

        if parity in cp_to_vec:
            cp_to_vec[parity].append(point)
        else:
            cp_to_vec[parity] = [point]

    # Print elements in each bin
    # for key, val in cp_to_vec.items():
    #     print(key + ':')
    #     for pt in val:
    #         print(pt)

    pairs = []
    new_points = []

    # Naively select pairs within each coset
    for _, val in cp_to_vec.items():
        while len(val) > 1:
            p1 = val.pop(0)
            p2 = val.pop(0)
            pair = (p1, p2)
            pairs.append(pair)

        # optionally keep unpaired vectors
        # new_points += val

    # Get the average of each pair
    for p1, p2 in pairs:
        # print(p1, p2)
        new_pt = avg(p1, p2)
        new_pt_2 = diff_avg(p1, p2)
        # print(new_pt)
        new_points.append(new_pt)
        new_points.append(new_pt_2)

    return new_points


def as_18(dim=2, N=1000, max_iters=50) -> LatticePoint:
    # lattice = IntegerLattice(dim)
    basis = np.array([[1, 2, 3], [4, 5, 6], [7, 8, -9]])
    lattice = RealLattice(basis)

    points = [lattice.sample_dgd() for _ in range(N)]
    print('Points:\n', [point for point in points[:10]])

    # Perform iterations
    for _ in range(max_iters):
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
    shortest_point = as_18(dim=10, N=10000, max_iters=1000)
    print('Shortest point:', shortest_point)
    print('Took', time() - start, 'seconds')
