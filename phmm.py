import random

import numpy as np
import scipy.optimize as opt

import util


class FixedHMM(util.Logged):
    def __init__(self, pZ, pE, pT, logger='default'):
        """Intialize a HMM with fixed parameters.

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
        self.S, self.A = pE.shape

        self.pZ = pZ
        self.pE = pE
        self.pT = pT

        super(FixedHMM, self).__init__(logger)

    @property
    def pZ(self):
        return self._pZ

    @pZ.setter
    def pZ(self, value):
        assert value.shape == (self.S,)

        self._pZ = value
        self.dist_z = util.Categorical(self.pZ)

    @property
    def pT(self):
        return self._pT

    @pT.setter
    def pT(self, value):
        assert value.shape == (self.S, self.S)

        self._pT = value
        self.logPT = np.log(np.where(self._pT>0, self._pT, 1e-100))
        self.dist_t = [util.Categorical(p) for p in self._pT]

    @property
    def pE(self):
        return self._pE

    @pE.setter
    def pE(self, value):
        assert value.shape == (self.S, self.A)

        self._pE = value
        self.logPE = np.log(np.where(self._pE>0, self._pE, 1e-100))
        self.dist_e = [util.Categorical(p) for p in self._pE]

    def generate(self, n):
        """Generate a random sequence of length n"""
        X = np.zeros((n,), dtype=int)
        Z = np.zeros((n,), dtype=int)

        Z[0] = self.dist_z.sample()
        X[0] = self.dist_e[Z[0]].sample()
        for i in xrange(1, n):
            Z[i] = self.dist_t[Z[i-1]].sample()
            X[i] = self.dist_e[Z[i-1]].sample()
        return X, Z


    def forward_backward(self, X):
        """Run the forward-backwards algorithm using data X.

        Compute the following magnitudes using the current parameters:

            - P(X[:i+1]          , Z[i]=j         )      alpha [j,    i]
            - P(X[i:  ]                   | Z[i]=j)      beta  [j,    i]
            - P(         Z[i-1]=j, Z[i]=k | X     )      xi    [j, k, i]
            - P(         Z[i  ]=j         | X     )      gamma [j,    i]

            - sum E   [log P(X,Z)]                       logE
                  Z|X
        """

        n = len(X)

        self.scale = np.zeros((n, ))

        # alpha[j, i] = P(X[:i+1], Z[i]=j)
        self.alpha = np.zeros((self.S, n))
        self.alpha[:, 0] = self.pE[:, X[0]] * self.pZ
        self.scale[0] = 1.0/self.alpha[:, 0].sum()
        self.alpha[:, 0] *= self.scale[0]
        for i in xrange(1, n):
            self.alpha[:, i] = self.pT.T.dot(self.alpha[:, i - 1])*self.pE[:,X[i]]
            self.scale[i] = 1.0/(self.alpha[:, i].sum())
            self.alpha[:, i] *= self.scale[i]

        # beta[j, i] = P(X[i:] | Z[i]=j)
        self.beta = np.zeros((self.S, n))
        self.beta[:, n-1] = self.pE[:, X[n-1]]*self.scale[n-1]
        for i in xrange(n - 1, 0, -1):
            self.beta[:, i - 1] = self.scale[i - 1]*self.pT.dot(self.beta[:, i])*self.pE[:, X[i - 1]]

        # xi[j, k, i] = P(Z[i-1]=j, Z[i]=k | X)
        self.xi = np.zeros((self.S, self.S, n))
        for i in xrange(1, n):
            self.xi[:, :, i] = (
                self.alpha[:, i - 1].reshape((self.S,1)) *
                self.beta[:, i]) * self.pT
            self.xi[:, :, i] /= self.xi[:, :, i].sum()

        # gamma[j, i] = P(Z[i]=j | X)
        self.gamma = self.xi.sum(axis=0)
        self.gamma[:, 0] = self.xi[:, :, 1].sum(axis=1)

        #  sum E   [log P(X,Z)]
        #       Z|X
        self.logE = self.gamma[:, 0].dot(self.logPE[:, X[0]])
        for i in xrange(1, n):
            self.logE += self.gamma[:,i].dot(self.logPE[:,X[i]])
            self.logE += np.sum(self.xi[:, :, i]*self.logPT)



class ProfileHMM(FixedHMM):
    def __init__(self, f, t, eps=None, p0=None):
        """Initialize a Profile HMM with.

        For an alphabet of size A, the following parameters:
           - f: matrix (W+1, A)
                f[0 , :] Background emission probabilities
                f[1:, :] Motif emission probabilities

           - t: vector (5, )
                t = [b, d, mb, m, md]
                    - b : probability of staying at background state

                    - d : probability of staying at a delete state

                    - mb: probability of switching from motif to background
                    - m : probability of staying at motid
                    - md: probability of switching from motif to delete state

                          mb + m + md = 1
        """
        self._f = f
        self._t = t

        self.W = f.shape[0] - 1 # motif length
        self.A = f.shape[1]     # alphabet size
        self.S = 2*self.W       # number of states

        self.eps = eps if eps is not None else np.repeat(1.0, self.A)
        self.p0 = p0 if p0 is not None else np.repeat(1.0/self.S, self.S)

        # Initialize FixedHMM
        super(ProfileHMM, self).__init__(self.p0, self.calc_pE(f), self.calc_pT(t))


    def calc_pE(self, f):
        pE = np.zeros((self.S, self.A))
        pE[:self.W, :] = self.f[ 0, :] # copy W times the background emissions
        pE[self.W:, :] = self.f[1:, :] # copy as is the motif probabilities
        return pE


    def calc_pT(self, t):
        b, d, mb, m, md = t
        pT = np.zeros((self.S, self.S))
        F = md*(d**np.arange(self.W))*(1-d)/(1 - d**self.W)
        for j in xrange(self.W):
            pT[j, j         ] = b     # b[j] -> b[j]
            pT[j, self.W + j] = 1 - b # b[j] -> m[j]

            pT[self.W + j, (j + 1) % self.W] = mb    # m[j] -> b[j + 1]
            for k in xrange(self.W):
                pT[self.W + j, self.W + k] = F[(k - j - 2) % self.W]
            pT[self.W + j, self.W + (j + 1) % self.W] += m
        return pT


    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = value
        self.pT = self.calc_pT(value)

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self._f = value
        self.pE = self.calc_pE(value)

    def fit_em(self, sequence, precision=1e-5, max_iter=100):
        n = len(sequence)

        it = 0
        logE = None
        while True:
            self.forward_backward(sequence)
            self.logE += np.sum((self.eps - 1)*self.f[1:,:])

            f = np.zeros(self.f.shape)
            fB = f[ 0, :]
            fM = f[1:, :]
            for i in xrange(n):
                fB[   X[i]] += self.gamma[:self.W, i].sum()
                fM[:, X[i]] += self.gamma[self.W:, i]

            fB /= fB.sum()
            fM[:] = util.normalized(fM + self.eps - 1)

            # count state transitions
            c = self.xi.sum(axis=2)

            b1 = 0.0
            b2 = 0.0
            for j in xrange(self.W):
                b1 += c[j,          j] # stay at background
                b2 += c[j, self.W + j] # switch from background to motif
            b = b1/(b1 + b2)

            k0 = 0.0
            k1 = np.zeros((self.W,))
            for j in xrange(self.W):
                k0 += c[self.W + j, (j + 1) % self.W]
            for l in xrange(self.W):
                for j in xrange(self.W):
                    k1[l] += c[self.W + j, self.W + (j + l + 2) % self.W]

            # objective function for parameters
            def g(tM):
                d, mb, m, md = tM
                s = md*(d**np.arange(self.W))*(1 - d)/(1 - d**self.W)
                s[self.W - 1] += m
                r = -(k0*np.log(mb) + k1.dot(np.log(s)))
                return r

            prange = (1e-6, 1.0 - 1e-6)
            res = opt.fmin_slsqp(
                func        = g,
                x0          = self.t[1:],
                bounds      = 4*[prange],
                eqcons      = [lambda x: x[1:].sum() - 1.0],
                iprint      = 0
            )
            d, mb, m, md = res

            # Check convergence
            if logE:
                err = np.abs(self.logE - logE)/np.abs(logE)
                if err < precision:
                    break

            logE = self.logE
            it += 1
            if it > max_iter:
                self.logger.warning(
                    'ProfileHMM.fit_em: max iterations reached without convergence (err={0})'.format(err))
                break

            self.f = f
            self.t = np.array([b, d, mb, m, md])

    @classmethod
    def fit(cls, sequence, window_min, window_max=None,
            gamma=0.1, n_seeds=1, precision=1e-5, max_iter=200):
        if window_max is None:
            window_max = window_min

        n = len(sequence)

        code_book = util.CodeBook(sequence)
        A = len(code_book)
        X = np.array(map(code_book.code, sequence))

        W0 = 1
        logP0 = np.sum(
            code_book.frequencies*np.log(code_book.frequencies))
        G = None
        best_W = None

        for W in xrange(window_min, window_max + 1):
            logE = None
            best_seed = None
            for b in np.linspace(0, 1, 5, endpoint=False)[1:]:
                for d in np.linspace(0, 1, 5, endpoint=False)[1:]:
                    for s in xrange(n_seeds):
                        # pick a random subsequence
                        i = random.randint(0, len(X) - W - 1)
                        f0 = util.guess_emissions(code_book, X[i:i + W])
                        t0 = np.random.rand(5)
                        t0[0] = b
                        t0[1] = d
                        t0[2:] /= t0[2:].sum()
                        eps = 1.0 + 1e-6*f0[0,:]
                        p0 = np.repeat(1.0/(2*W), 2*W)

                        hmm = cls(f=f0, t=t0, p0=p0, eps=eps)
                        hmm.fit_em(X, precision, max_iter)

                        if logE is None or hmm.logE > logE:
                            logE = hmm.logE
                            best_seed = hmm

            G2 = util.model_score(n*logP0, logE, (W - W0)*(A - 1))
            hmm.logger.info(
                'ProfileHMM.fit E1={0} G={1}'.format(logE, G2))

            if G is None or G2 < G:
                G = G2
                best_W = best_seed

            if G == 0:
                hmm.logger.info('Switched null model (W={0})'.format(W))
                G = 1.0
                logP0 = logE/n
                W0 = float(W)

        return best_W


if __name__ == '__main__':
    phmm_true = ProfileHMM(
        f=np.array([
            [0.2, 0.3, 0.2, 0.3],
            [0.9, 0.1, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2]]),
        t = np.array([
            0.8, 0.05, 0.05, 0.9, 0.05])
    )

    X, Z = phmm_true.generate(1000)
    phmm = ProfileHMM.fit(X, 2, 4)
    print phmm_true.f
    print phmm.f
    print phmm_true.t
    print phmm.t
