import numpy as np

import phmm
import util

def test_phmm_1():
    W = 3    
    A = 4
    f1 = util.normalized(np.random.rand(W+1, A))
    t1 = np.random.rand(6)
    t1[2:] /= t1[2:].sum()
    H1 = phmm.ProfileHMM(f1, t1)

    X, Z = H1.generate(10000)

    f2 = util.normalized(f1 + np.random.rand(W+1, A))
    t2 = t1.copy()
    t2[:2] = np.random.rand(2)
    t2[2:] = np.random.rand(4)
    t2[2:] /= t2[2:].sum()
    H2 = phmm.ProfileHMM(f2, t2)

    H2.fit_em(X, precision=1e-5, max_iter=1000)

    print H2.f
    print H2.t
    print H1.f
    print H1.t

if __name__ == '__main__':
    test_phmm_1()
