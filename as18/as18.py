from lattice import Lattice, LatticePoint, IntegerLattice
import numpy as np
from numpy import linalg as LA

def avg(p1, p2):
    # Take the average of two lattice points, provided that they are in the same
    # coset
    coords = np.mean([p1.coords, p2.coords], axis=0)
    pt = LatticePoint(p1.lattice, coords)
    return pt

def diff_avg(p1, p2):
    # Take the difference average
    coords = (p1.coords - p2.coords) / 2
    pt = LatticePoint(p1.lattice, coords)
    return pt

def shortest(points):
    # Find the shortest vector
    shortest_vec = None
    dim = points[0].vec.size
    for point in points:
        # print(point)
        if (point.vec != np.zeros(dim)).any():
            shortest_vec = point
        elif shortest_vec is not None:
            if point.norm < shortest_vec.norm:
                shortest_vec = point
    return shortest_vec

def iter(points):
    # Perform one iteration
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

# Initial parameters
# basis = np.array([[1, 0], [0, 1]])
# lattice = Lattice(basis)
dim = 2
lattice = IntegerLattice(dim)
N = 1000
iters = 50

### Full algorithm

points = []
print('Points:')
# Generate uniform distribution of points with coordinates less than 1000
for i in range(N):
    point = lattice.sample_dgd()
    print(point)
    points.append(point)

# Perform iterations
for i in range(iters):
    new_points = iter(points)

    new_points_norms = [LA.norm(pt.vec) ** 2 for pt in new_points]
    print('avg of sq norms:', np.mean(new_points_norms))
    # print('New Points:')
    # for point in new_points:
    #     print(point)

    if np.mean(new_points_norms) == 0:
        break

    points = new_points

print(shortest(points))
