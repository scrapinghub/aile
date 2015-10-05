import numpy as np

import hmm
import util


def test_hmm():
    pE = np.array([
            [0.4, 0.2, 0.4],
            [0.0, 0.9, 0.1],
            [0.1, 0.1, 0.8]
    ])
    pT = np.array([
        [0.5 ,  0.5, 0.0 ],
        [0.0 , 0.25, 0.75],
        [0.75, 0.25, 0.0 ]
    ])

    H = hmm.FixedHMM(
        pZ = np.repeat(1.0/3.0, 3),
        pE = pE,
        pT = pT
    )

    pS = pE.T.dot(
        np.linalg.matrix_power(pT, 1000)[0,:])
    n = 100000
    X, Z = H.generate(n)
    f = np.bincount(X).astype(float)
    f /= f.sum()
    assert util.eq_delta(pS, f, 1e-2)

    H.forward_backward(X)

    xi1 = np.zeros(pT.shape)
    for i in xrange(n - 1):
        xi1[Z[i], Z[i+1]] += 1.0
    xi1 /= xi1.sum()

    xi2 = H.xi.sum(axis=2)
    xi2 /= xi2.sum()

    assert util.eq_delta(xi1, xi2, 1e-2)

    gamma1 = np.zeros(pT.shape[0])
    for i in xrange(n):
        gamma1[Z[i]] += 1.0
    gamma1 /= gamma1.sum()

    gamma2 = H.gamma.sum(axis=1)
    gamma2 /= gamma2.sum()

    assert util.eq_delta(gamma1, gamma2, 1e-2)
