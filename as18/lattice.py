import numpy as np
from random import randint

class Lattice:
    def __init__(self, basis):
        self.basis = basis
        self.dim = basis.shape[0]

    def sample_uniform(self, max_digits=3):
        coords = np.random.rand(self.dim)
        coords = np.round(coords * 10**max_digits)
        coords -= (10**max_digits)/2
        coords = coords.astype(int)

        return LatticePoint(self, coords)

class IntegerLattice(Lattice):
    def __init__(self, dim=2):
        self.basis = np.identity(dim)
        self.dim = dim

    def sample_dgd(self, s=10, t=5):
        coords = np.array([self._sample_dgd_single(s, t) for _ in range(self.dim)])
        return LatticePoint(self, coords)

    def _sample_dgd_single(self, s=10, t=5):
        while True:
            z = randint(-t*s, t*s)
            if np.exp(-np.pi * z**2 / s**2) <= np.random.rand():
                break
        return z


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