import numpy as np
from as18.lattice import RealLattice, LatticePoint

def test_lattice():
    basis = np.array([[2, -1], [1, 1]])
    lp1 = LatticePoint(basis=basis, coords=np.array([1, 1]))
    lp2 = LatticePoint(basis=basis, vec=np.array([3, 0]))
    assert lp1 == lp2
