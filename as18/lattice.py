import numpy as np
from random import randint, random
from abc import ABC

def lin_ind(basis: np.ndarray) -> bool:
    """Returns whether the input basis is linearly independent. (Unimplemented)"""
    return False

class LatticePoint:
    # TODO: Allow instantiation from just a vector and a basis, ensuring that
    # that the vector actually is in the lattice.
    def __init__(self, basis: np.ndarray, coords: np.ndarray):
        self.basis = basis
        self.coords = coords
        self.dim = coords.size

    def __repr__(self):
        return str(self.vec)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return np.array_equal(self.vec, other.vec)

    @property
    def vec(self) -> np.ndarray:
        return self.basis @ self.coords

    @property
    def norm(self) -> float:
        return np.linalg.norm(self.vec)

class Lattice(ABC):
    """An abstract class representing a lattice."""

    # TODO: Ensure that the basis is linearly independent and is square.
    def __init__(self, basis: np.ndarray):
        """Instantiate a Lattice given by a [basis]."""
        self.basis = basis
        self.dim = basis.shape[0]

    def sample_uniform(self, min_max=500) -> LatticePoint:
        """Uniformly sample a point from a finite range of the lattice.

        Args:
            min_max (int, optional): Minimum and maximum coordinate to sample
                from. Defaults to 500.

        Returns:
            LatticePoint: Uniformly sampled lattice point.
        """
        coords = [randint(-min_max, min_max) for _ in range(self.dim)]
        coords_np = np.array(coords)

        return LatticePoint(self, coords_np)

class IntegerLattice(Lattice):
    """The Z^n lattice."""

    def __init__(self, dim=2):
        """Instantiate a [dim]-dimensional integer lattice. Defaults to Z^2."""
        self.basis = np.identity(dim)
        self.dim = dim

    def sample_dgd(self, s=10, t=5) -> LatticePoint:
        """Sample a point from this lattice according to the Discrete Gaussian
        Distribution.

        Args:
            s (int, optional): The "standard deviation" of the Discrete
                Gaussian. Defaults to 10.
            t (int, optional): The sampling scale factor. Defaults to 5.
        """
        coords = [self._sample_dgd_single(s, t) for _ in range(self.dim)]
        coords_np = np.array(coords)

        return LatticePoint(self.basis, coords_np)

    def _sample_dgd_single(self, s=10, t=5) -> int:
        """Sample a single coordinate for the Discrete Gaussian."""
        while True:
            z = randint(-t*s, t*s)
            if np.exp(-np.pi * z**2 / s**2) <= random():
                break
    
        return z
