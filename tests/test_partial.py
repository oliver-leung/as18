import numpy as np

from as18.lattice import RealLattice
from partial import partial


def test_partial_real_lat():
    basis = np.array([
        [1, 5, 4, -2],
        [4, -1, 5, -3],
        [-3, 7, 10, -9],
        [0, 6, -2, 14]
    ])
    lat = RealLattice(basis)
    shortest_vec = partial(lat)
    print('Shortest point from partial binning:', shortest_vec)
