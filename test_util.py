import numpy as np

import util


def test_categorical():
    p = np.array([0.25, 0.5, 0.25])
    cat = util.Categorical(p)
    X = cat.sample(1000000)
    f = np.bincount(X).astype(float)
    f /= f.sum()
    assert util.eq_delta(f, p, 1e-2)
