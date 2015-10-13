import itertools
import time

import numpy as np
import pomegranate

import hmm
import util


def test_hmm_1():
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

    fb = H.forward_backward(X)

    xi1 = np.zeros(pT.shape)
    for i in xrange(n - 1):
        xi1[Z[i], Z[i+1]] += 1.0
    xi1 /= xi1.sum()

    xi2 = fb.xi
    xi2 /= xi2.sum()

    assert util.eq_delta(xi1, xi2, 1e-2)

    gamma1 = np.zeros(pT.shape[0])
    for i in xrange(n):
        gamma1[Z[i]] += 1.0
    gamma1 /= gamma1.sum()

    gamma2 = fb.gamma.sum(axis=1)
    gamma2 /= gamma2.sum()

    assert util.eq_delta(gamma1, gamma2, 1e-2)


def test_hmm_2():
    pZ = np.repeat(1.0/3.0, 3)
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
        pZ = pZ,
        pE = pE,
        pT = pT
    )

    n = 100000
    X, Z = H.generate(n)

    print 'test_hmm_2'
    print '----------'
    t1 = time.clock()
    fb = H.forward_backward(X)
    t2 = time.clock()
    print 'aile        FB: {0:.5f}s'.format(t2 - t1)
    G = pomegranate.HiddenMarkovModel("test")
    states = [
        pomegranate.State(
            pomegranate.DiscreteDistribution(dict(enumerate(p))),
            name=str(s))
        for s, p in enumerate(pE)]
    G.add_states(states)

    for s, p in enumerate(pZ):
        G.add_transition(G.start, states[s], p)

    for s, p in enumerate(pT):
        for t, q in enumerate(p):
            G.add_transition(states[s], states[t], q)

    G.bake()
    t1 = time.clock()
    logT, DP = G.forward_backward(X)
    t2 = time.clock()
    print 'pomegranate FB: {0:.5f}s'.format(t2 - t1)

    g1 = fb.gamma.sum(axis=1)
    g2 = np.exp(DP).sum(axis=0)
    g1 /= g1.sum()
    g2 /= g2.sum()
    assert min(
            np.abs(g1[p,] - g2).sum()
            for p in itertools.permutations(range(3))) < 1e-3

    assert util.eq_relative(fb.logP, G.log_probability(X))


def test_hmm_3():
    A = 20
    S = 160
    n = 4000
    N = 10

    pZ = np.repeat(1.0/S, S)
    pE = util.normalized(np.random.rand(S, A))
    pT = util.normalized(np.random.rand(S, S))

    H = hmm.FixedHMM(
         pZ = pZ,
         pE = pE,
         pT = pT
     )

    X, Z = H.generate(n)
    print 'test_hmm_3'
    print '----------'
    print '    A = {0}'.format(A)
    print '    S = {0}'.format(S)
    print '    n = {0}'.format(n)

    for i in range(N):
        H = hmm.FixedHMM(
             pZ = pZ,
             pE = util.normalized(np.random.rand(S, A)),
             pT = util.normalized(np.random.rand(S, S))
        )
        t1 = time.clock()
        H.forward_backward(X)
        t2 = time.clock()
        print '    {0:2d}/{1:2d} FB: {2:.5f}s'.format(i + 1, N, t2 - t1)

if __name__ == '__main__':
    test_hmm_1()
    test_hmm_2()
    test_hmm_3()
