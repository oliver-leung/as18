import numpy as np

class Lattice:
    def __init__(self, basis):
        self.basis = basis
        self.dim = basis.shape[0]

    def generate(self, max_digits=3):
        coords = np.random.rand(self.dim)
        coords = np.round(coords * 10**max_digits)
        coords -= (10**max_digits)/2
        coords = coords.astype(int)

        return LatticePoint(self, coords)

class LatticePoint:
    def __init__(self, lattice, coords):
        self.lattice = lattice
        self.coords = coords

    def __str__(self):
        return str(self.vec)

    def __eq__(self, other):
        return self.coords == other.coords

    @property
    def vec(self):
        return self.lattice.basis @ self.coords

    @property
    def norm(self):
        return np.linalg.norm(self.vec)