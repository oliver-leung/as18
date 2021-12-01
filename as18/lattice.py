import abc
from collections import defaultdict
from math import ceil, floor
from typing import List, Dict

import numpy as np
from random import randint, random
from abc import ABC

from matplotlib import pyplot as plt

from utility import gram_schmidt, validate_basis, gaussian


class LatticePoint:
    def __init__(self, basis: np.ndarray, coords: np.ndarray = None, vec: np.ndarray = None):
        if coords is None and vec is None:
            raise ValueError("Either coords or vec must be specified.")
        if coords is not None and vec is not None:
            raise ValueError("Only one of coords or vec can be specified.")

        self.basis = basis
        self.dim = basis.shape[0]

        if coords is not None:
            self.coords = coords
        elif vec is not None:
            self.coords = np.linalg.solve(self.basis.T, vec)

        coords_int = np.rint(self.coords)
        if not np.isclose(self.coords, coords_int).all():
            raise ValueError("Vector is not in the lattice.")

        self.coords = coords_int

    @classmethod
    def from_coords(cls, basis, coords):
        return cls(basis, coords=coords)

    @classmethod
    def from_vec(cls, basis, vec):
        return cls(basis, vec=vec)

    def __repr__(self):
        return str(self.vec)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return np.array_equal(self.vec, other.vec)

    def __add__(self, other):
        assert np.array_equal(self.basis, other.basis)
        coords = self.coords + other.coords
        return LatticePoint(self.basis, coords)

    def __sub__(self, other):
        assert np.array_equal(self.basis, other.basis)
        coords = self.coords - other.coords
        return LatticePoint(self.basis, coords)

    def __neg__(self):
        return LatticePoint(self.basis, -self.coords)

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            coords = self.coords / other
            return LatticePoint(self.basis, coords)

        elif isinstance(other, LatticePoint):
            raise ValueError("Can't divide two LatticePoints")
        else:
            raise ValueError

    @property
    def vec(self) -> np.ndarray:
        return self.basis.T @ self.coords

    @property
    def norm(self) -> float:
        return np.linalg.norm(self.vec)

    @property
    def is_zero(self) -> bool:
        zeros = np.zeros(self.dim)
        return np.array_equal(self.vec, zeros)


class Lattice(ABC):
    """An abstract class representing a lattice."""

    def __init__(self, basis: np.ndarray):
        """Instantiate a Lattice given by a [basis]."""
        validate_basis(basis)

        self.basis = basis
        self.dim = basis.shape[0]

    def sample_uniform(self, min_max=500) -> LatticePoint:
        """Uniformly sample a point from a finite range of coordinates.

        Args:
            min_max (int, optional): Minimum and maximum coordinate to sample
                from. Defaults to 500.

        Returns:
            LatticePoint: Uniformly sampled lattice point.
        """
        coords = [randint(-min_max, min_max) for _ in range(self.dim)]
        coords_np = np.array(coords)

        return LatticePoint(basis=self.basis, coords=coords_np)

    @abc.abstractmethod
    def sample_dgd(self) -> LatticePoint:
        raise NotImplementedError("Can't sample from DGD over this lattice")


class IntegerLattice(Lattice):
    """The Z^n lattice."""

    def __init__(self, dim=2):
        """Instantiate a [dim]-dimensional integer lattice. Defaults to Z^2."""
        super().__init__(np.identity(dim))

    def sample_dgd(self, s=10, t=5) -> LatticePoint:
        """Sample a point from this lattice according to the Discrete Gaussian
        Distribution.

        Args:
            s (int, optional): The "standard deviation" of the Discrete
                Gaussian. Defaults to 10.
            t (int, optional): The sampling scale factor. Defaults to 5.
        """
        coords = [sample_dgd_Z(s, t) for _ in range(self.dim)]
        coords_np = np.array(coords)

        return LatticePoint(basis=self.basis, coords=coords_np)


class RealLattice(Lattice):
    """The R^n lattice."""

    def sample_dgd(self, s=10, c: np.ndarray = None) -> LatticePoint:
        """Sample a point from this lattice according to the Discrete Gaussian
        Distribution.

        Args:
            s (int, optional): The "standard deviation" of the Discrete
                Gaussian. Defaults to 10.
            c: (np.ndarray, optional):
        """
        if not c:
            c = np.zeros(self.dim)

        B_t = gram_schmidt(self.basis)
        z_p = np.zeros(self.dim)

        for i in reversed(range(self.dim)):
            c_p = np.dot(c, B_t[i]) / np.dot(B_t[i], B_t[i])
            s_p = s / np.linalg.norm(B_t[i])

            z_p[i] = sample_dgd_Z(s=s_p, c=c_p)
            c = c - (z_p[i] * self.basis[i])

        return LatticePoint(basis=self.basis, coords=z_p)


class QaryLattice(RealLattice):
    """A lattice subset of Z^n; For some integer vector a in Z^n mod q, the q-ary lattice is the set of all points
    such that the inner product with a is zero. Note that we fix a_n = 1 in this implementation.
    """

    def __init__(self, q: int = 10 ** 6, dim: int = None, a: np.ndarray = None):
        if dim and a:
            raise ValueError('Only one of dim and a may be specified.')
        elif not dim and not a:
            raise ValueError('Either dim or a must be specified.')
        elif dim and not a:
            a = np.random.randint(0, q - 1, size=dim - 1)
        elif not dim and a:
            dim = a.shape[0] + 1

        a = np.append(a, q)
        basis = np.identity(dim)
        basis[-1] = a
        basis = basis.T
        self.q = q

        super().__init__(basis)

    def sample_dgd(self, s=10, c: np.ndarray = None) -> LatticePoint:
        if s is None:
            s = 10 * self.q
        return super().sample_dgd(s, c)


def sample_dgd_Z(s: float = 10, t: float = 5, c: float = 0) -> int:
    """
    Sample from the discrete Gaussian over some integer range (i.e. the Z lattice).

    Args:
        s (float, optional): The Gaussian parameter (i.e. standard deviation).
        t (float, optional): The scaling factor, which determines the range of sampling.
        c (float, optional): The center around which to sample the discrete Gaussian.
    """
    while True:
        # Randomly sample an integer from the range. Note that we round the range "outwardly".
        max_int = ceil(c + t * s)
        min_int = floor(c - t * s)
        z = randint(min_int, max_int)

        # Accept the sampled integer with probability rho_s(z - c)
        # if random() <= np.exp(-np.pi * (z - c) ** 2 / s ** 2):
        rand_num = random()
        rho = gaussian(s, z - c)
        if rand_num <= rho:
            break

    return z


def shortest(points: List[LatticePoint], with_zeros=False) -> LatticePoint:
    """Find the shortest point within a list of points with the same dimensionality."""
    if not with_zeros:
        points = remove_zeros(points)

    return min(points, key=lambda x: x.norm)  # Finds the minimum according to the norm


def remove_zeros(points: List[LatticePoint]) -> List[LatticePoint]:
    zeros = np.zeros(points[0].dim)

    return [pt for pt in points if not np.array_equal(pt.vec, zeros)]


def visualize(lattice_pts: List[LatticePoint], noisy=True) -> None:
    vectors = np.array([pt.vec for pt in lattice_pts]).T
    if noisy:
        vectors = np.array([vec + np.random.normal(0, 0.1, len(lattice_pts)) for vec in vectors])

    fig = plt.figure(num=0)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*(vectors[:3]))  # Plot the first 3 dimensions, in case the points are in higher dimensions
    plt.pause(0.5)


def bin_by_coset(points: List[LatticePoint], start: int = None, end: int = None, mod: int = 2)\
        -> Dict[str, List[LatticePoint]]:

    if start is None:
        start = 0
    if end is None:
        if not points:  # Avoid case where points is empty
            return {}
        end = points[0].dim

    coset_to_vecs = defaultdict(list)  # Mapping from cosets to vector lists

    # Bin according to mod 2L
    for point in points:
        if start < end:
            parity_coords = point.coords[start: end]
        else:
            parity_coords = point.coords.take(range(start, end + 4), mode='wrap')

        coords_str = (parity_coords % mod).astype(int).astype(str)
        parity = ''.join(coords_str)

        coset_to_vecs[parity].append(point)

    return coset_to_vecs
