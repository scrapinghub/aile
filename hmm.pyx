import numpy as np

import util

cimport numpy as np
cimport cython


class FixedHMM(util.Logged):
    def __init__(self,
                 np.ndarray[np.double_t, ndim=1] pZ,
                 np.ndarray[np.double_t, ndim=2] pE,
                 np.ndarray[np.double_t, ndim=2] pT,
                 logger='default'):
        """Initialize a HMM with fixed parameters.

        - pZ: prior on the first hidden state Z1
              pZ[s] = Probability of being at state 's'

        - pE: probability of emissions
              pZ[s, a] = Probability of emitting symbol 'a' from state 's'

        - pT: probability of transitions
              pT[s1, s2] = Probability of going from state s1 to s2

        We assume there are S hidden states and A different emission symbols.
        The shapes of the arrays must be then:

            pZ: (S,)
            pE: (S, A)
            pT: (S, S)
        """
        self.S = pE.shape[0]
        self.A = pE.shape[1]

        self.pZ = pZ
        self.pE = pE
        self.pT = pT

        super(FixedHMM, self).__init__(logger)

    @property
    def pZ(self):
        return self._pZ

    @pZ.setter
    def pZ(self, np.ndarray value):
        self._pZ = value
        self.logPZ = util.safe_log(self._pZ)
        self.dist_z = util.Categorical(self.pZ)

    @property
    def pT(self):
        return self._pT

    @pT.setter
    def pT(self, value):
        self._pT = value
        self.logPT = util.safe_log(self._pT)
        self.dist_t = [util.Categorical(p) for p in self._pT]

    @property
    def pE(self):
        return self._pE

    @pE.setter
    def pE(self, value):
        self._pE = value
        self.logPE = util.safe_log(self._pE)
        self.dist_e = [util.Categorical(p) for p in self._pE]

    def generate(self, int n):
        """Generate a random sequence of length n"""
        cdef np.ndarray[np.int_t, ndim=1] X = np.zeros((n,), dtype=int)
        cdef np.ndarray[np.int_t, ndim=1] Z = np.zeros((n,), dtype=int)

        cdef int i

        Z[0] = self.dist_z.sample()
        X[0] = self.dist_e[Z[0]].sample()
        for i in range(1, n):
            Z[i] = self.dist_t[Z[i-1]].sample()
            X[i] = self.dist_e[Z[i  ]].sample()
        return X, Z

    @cython.boundscheck(False)
    def forward_backward(self, np.ndarray[np.int_t, ndim=1] X):
        """Run the forward-backwards algorithm using data X.

        Compute the following magnitudes using the current parameters:

            - P(X[:i+1]          , Z[i]=j         )      alpha [j,    i]
            - P(X[i:  ]                   | Z[i]=j)      beta  [j,    i]
            - P(         Z[i-1]=j, Z[i]=k | X     )      xi    [j, k, i]
            - P(         Z[i  ]=j         | X     )      gamma [j,    i]

            - sum E   [log P(X,Z)]                       logE
                  Z|X
        """
        cdef unsigned int S = self.S

        cdef np.ndarray[np.double_t, ndim=1] pZ = self.pZ
        cdef np.ndarray[np.double_t, ndim=2] pE = self.pE
        cdef np.ndarray[np.double_t, ndim=2] pT = self.pT
        cdef np.ndarray[np.double_t, ndim=2] logPT = self.logPT
        cdef np.ndarray[np.double_t, ndim=2] logPE = self.logPE

        cdef unsigned int n = len(X) # sequence length
        cdef unsigned int i          # index from 0 to n - 1
        cdef double a, b             # accumulators
        cdef unsigned int s, t       # index from 0 to S - 1

        # to avoid numerical precision errors we multiply alpha and beta
        # by this number
        cdef np.ndarray[np.double_t, ndim=1] scale = np.zeros((n, ))

        # alpha[j, i] = P(X[:i+1], Z[i]=j)
        cdef np.ndarray[np.double_t, ndim=2] alpha = np.zeros((S, n))

        a = 0
        b = 0
        for s in range(S):
            b = alpha[s, 0] = pE[s, X[0]]*pZ[s]
            a += b
        a = 1.0/a
        for s in range(S):
            alpha[s, 0] *= a
        scale[0] = a

        for i in range(1, n):
            a = 0
            for s in range(S):
                b = 0
                for t in range(S):
                    b += pT[t, s]*alpha[t, i-1]
                b *= pE[s, X[i]]
                a += b
                alpha[s, i] = b
            if a > 0:
                a = 1.0/a
            for s in range(S):
                alpha[s, i] *= a
            scale[i] = a

        # beta[j, i] = P(X[i:] | Z[i]=j)
        cdef np.ndarray[np.double_t, ndim=2] beta = np.zeros((S, n))
        for s in range(S):
            beta[s, n-1] = pE[s, X[n-1]]*scale[n-1]
        for i in range(n - 1, 0, -1):
            a = scale[i - 1]
            for s in range(S):
                b = 0
                for t in range(S):
                    b += pT[s, t]*beta[t, i]
                beta[s, i - 1] = b*a*pE[s, X[i - 1]]

        # xi[j, k, i] = P(Z[i-1]=j, Z[i]=k | X)
        cdef np.ndarray[np.double_t, ndim=3] xi = np.zeros((S, S, n))
        for i in range(1, n):
            a = 0
            for s in range(S):
                for t in range(S):
                    xi[s, t, i] = b = alpha[s, i - 1]*beta[t, i]*pT[s, t]
                    a += b
            for s in range(S):
                for t in range(S):
                    xi[s, t, i] /= a

        # gamma[j, i] = P(Z[i]=j | X)
        cdef np.ndarray[np.double_t, ndim=2] gamma = xi.sum(axis=0)
        gamma[:, 0] = xi[:, :, 1].sum(axis=1)

        # log P(X)
        self.scale = scale
        self.alpha = alpha
        self.beta = beta
        self.xi = xi
        self.gamma = gamma
        self.logP = -np.log(scale).sum()

    @cython.boundscheck(False)
    def viterbi(self, np.ndarray[np.int_t, ndim=1] X):
        cdef unsigned int S = self.S

        cdef np.ndarray[np.double_t, ndim=1] logPZ = self.logPZ
        cdef np.ndarray[np.double_t, ndim=2] logPT = self.logPT
        cdef np.ndarray[np.double_t, ndim=2] logPE = self.logPE

        cdef unsigned int n = len(X)

        cdef np.ndarray[np.double_t, ndim=2] delta = np.zeros((S, n))
        cdef np.ndarray[np.int_t, ndim=2] psi = np.zeros((S, n), dtype=np.int)

        cdef unsigned int i
        cdef unsigned int s, t, u
        cdef double a, b, c

        # log P(Z1, X1)
        for s in range(S):
            delta[s, 0] = logPZ[s] + logPE[s, X[0]]
        for i in range(1, n):
            for s in range(S):
                a = logPT[0, s] + delta[0, i - 1]
                u = 0
                for t in range(1, S):
                    b = logPT[t, s] + delta[t, i - 1]
                    if b > a:
                        u = t
                        a = b
                psi[s, i] = u
                delta[s, i] = a + logPE[s, X[i]]

        cdef np.ndarray[np.int_t, ndim=1] z = np.zeros((n,), dtype=np.int)

        a = delta[0, n - 1]
        u = 0
        for s in range(1, S):
            if delta[s, n - 1] > a:
                a = delta[s, n - 1]
                u = s
        z[n - 1] = u
        cdef double logP = delta[z[n - 1], n - 1]
        for i in range(n - 1, 0, -1):
            z[i - 1] = psi[z[i], i]

        return z, logP

    def score(self, np.ndarray[np.int, ndim=1] X, np.ndarray[np.int, ndim=1] Z):
        """Calculate log-probability of (X, Z)"""
        cdef np.ndarray[np.double_t, ndim=1] logPZ = self.logPZ
        cdef np.ndarray[np.double_t, ndim=2] logPT = self.logPT
        cdef np.ndarray[np.double_t, ndim=2] logPE = self.logPE

        cdef unsigned int n = len(X)

        cdef double logP = logPZ[Z[0]] + logPE[Z[0], X[0]]
        cdef unsigned int i
        for i in range(n):
            logP += logPT[Z[i-1], Z[i]] + logPE[Z[i], X[i]]
        return logP
