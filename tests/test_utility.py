import numpy as np
from as18.utility import gram_schmidt

def test_gram_schmidt():
    B = np.array([[1, 0], [2, 1]])
    gs = gram_schmidt(B)
    assert np.array_equal(gram_schmidt(B), [[1, 0], [0, 1]])
