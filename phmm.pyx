import math
import numpy as np
import scipy.optimize as opt

import util
import hmm

cimport numpy as np
cimport cython


class ProfileHMM(hmm.FixedHMM):
    def __init__(self, f, t, eps=None, p0=None):
        """Initialize a Profile HMM with.

        For an alphabet of size A, the following parameters:
           - f: matrix (W+1, A)
                f[0 , :] Background emission probabilities
                f[1:, :] Motif emission probabilities

           - t: vector (5, )
                t = [b_in, b_out, d, mb, m, md]
                    - b_in  : probability of staying at background state 2..W
                    - b_out : probability of staying at background state 1
                    - d     : probability of staying at a delete state

                    - mb    : probability of switching from motif to background
                    - m     : probability of staying at motid
                    - md    : probability of switching from motif to delete state

                          mb + m + md = 1
        """
        self._f = f
        self._t = t

        self.W = f.shape[0] - 1 # motif length
        self.A = f.shape[1]     # alphabet size
        self.S = 2*self.W       # number of states

        self.eps = eps if eps is not None else np.repeat(1.0, self.A)
        self.p0 = p0 if p0 is not None else np.repeat(1.0/self.S, self.S)

        self.code_book = None

        # Initialize hmm.FixedHMM
        super(ProfileHMM, self).__init__(self.p0, self.calc_pE(f), self.calc_pT(t))

    def calc_pE(self, f):
        pE = np.zeros((self.S, self.A))
        pE[:self.W, :] = self.f[ 0, :] # copy W times the background emissions
        pE[self.W:, :] = self.f[1:, :] # copy as is the motif probabilities
        return pE

    def calc_pT(self, t):
        b_in, b_out, d, mb, m, md = t
        pT = np.zeros((self.S, self.S))
        F = md*(d**np.arange(self.W))*(1-d)/(1 - d**self.W)
        F[self.W - 1] += m
        pT[0,      0] = b_out
        pT[0, self.W] = 1 - b_out
        for j in range(1, self.W):
            pT[j, j         ] = b_in     # b[j] -> b[j]
            pT[j, self.W + j] = 1 - b_in # b[j] -> m[j]
        for j in range(self.W):
            pT[self.W + j, (j + 1) % self.W] = mb    # m[j] -> b[j + 1]
            for k in range(self.W):
                pT[self.W + j, self.W + k] = F[(k - j - 2) % self.W]
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

    def fit_em_1(self, np.ndarray[np.int_t, ndim=1] sequence):
        self.forward_backward(sequence)
        # Take into account the priors on the parameters
        self.logE += np.sum((self.eps - 1)*self.f[1:,:])

        cdef unsigned int W = self.W
        cdef np.ndarray[np.double_t, ndim=2] gamma = self.gamma
        cdef np.ndarray[np.double_t, ndim=3] xi = self.xi

        cdef np.ndarray[np.int_t, ndim=1] X = sequence
        cdef unsigned int n = len(X)

        cdef unsigned int i, j, k, l

        cdef np.ndarray[np.double_t, ndim=2] f = np.zeros(self.f.shape)

        for i in range(n):
            for s in range(W):
                f[    0, X[i]] += gamma[    s, i]
                f[1 + s, X[i]] += gamma[W + s, i]

        f[0,  :] /= f[0, :].sum()
        f[1:, :] = util.normalized(f[1:, :] + self.eps - 1.0)

        # count state transitions
        cdef np.ndarray[np.double_t, ndim=2] c = xi.sum(axis=2)
        c /= c.sum()

        cdef double b1 = 0.0
        cdef double b2 = 0.0
        for j in range(1, W):
            b1 += c[j,     j] # stay at background
            b2 += c[j, W + j] # switch from background to motif
        cdef double b_in = b1/(b1 + b2)
        cdef double b_out = c[0, 0]/(c[0,0] + c[0, W])

        cdef np.ndarray[np.double_t, ndim=2] pT = self.pT

        # objective function for parameters
        def g(np.ndarray[np.double_t, ndim=1] tM,
              np.ndarray[np.double_t, ndim=2] c):
            cdef double d  = tM[0]
            cdef double mb = tM[1]
            cdef double m  = tM[2]
            cdef double md = tM[3]

            F = math.log(md) + np.log(d)*np.arange(self.W) + math.log(1-d) - math.log(1 - d**self.W)
            F[self.W - 1] = np.logaddexp(F[self.W - 1], math.log(m))
            r = 0
            for j in range(W):
                r -= c[W + j, (j + 1) % W]*math.log(mb)
                for k in range(W):
                    r -= c[W + j, W + k]*F[(k - j - 2) % W]

            return r

        d, mb, m, md = opt.fmin_slsqp(
            func        = g,
            x0          = self.t[2:].copy(),
            bounds      = 4*[(1e-3, 1.0 - 1e-3)],
            eqcons      = [lambda tM, c: tM[1:].sum() - 1.0],
            iprint      = 0,
            args        = (c/n, ),
            epsilon     = 1e-6,
            acc         = 1e-9
        )

        self.f = f
        self.t = np.array([b_in, b_out, d, mb, m, md])

    def fit_em(self,
               np.ndarray[np.int_t, ndim=1] sequence,
               double precision=1e-3,
               unsigned int max_iter=100):
        logE = None
        it = 0
        while True:
            self.fit_em_1(sequence)
            # Check convergence
            if logE:
                err = (logE - self.logE)/logE
                if err < 0:
                    self.logger.warning(
                        'ProfileHMM.fit_em: log(E) decreased {0}({1:3f}%)'.format(logE, err*100.0))
                if np.abs(err) < precision:
                    break
            logE = self.logE
            it += 1
            if it > max_iter:
                self.logger.warning(
                    'ProfileHMM.fit_em: max iterations reached without convergence (err={0})'.format(err))
                break

    @classmethod
    def fit(cls, sequence, window_min, window_max=None,
            gamma=0.1, steps=4, n_seeds=1, precision=1e-5,
            max_iter=200, guess_emissions=None, logger='default'):
        """Fit the model parameters using data.

        - sequence       : the sequence to fit
        - window_min     : minimum motif width to consider
        - window_max     : maximum motif width to consider.
                           If None only one motif width will be considered.
        - gamma          : parameter to pass to the guess_emissions function
        - steps          : number of steps in the grid (b_out,d)
        - n_seeds        : number of seeds per grid element
        - precision      : EM desired precision in logE
        - max_iter       : maximum iterations in EM
        - guess_emissions: a function to compute the initial value of the
                           emissions matrix. Signature:

                               guess_emissions(code_book, W, X)

                           where:
                                code_book: CodeBook used to encode X
                                W        : motif width
                                X        : encoded sequence
        """
        log = util.Logged(logger=logger)
        if window_max is None:
            log.logger.info('ProfileHMM.fit: using only one motif width (W = {0})'.format(window_min))
            window_max = window_min

        n = len(sequence)

        code_book = util.CodeBook(sequence)
        A = len(code_book)
        X = np.array(map(code_book.code, sequence))

        t_seeds = n_seeds*steps**2

        W0 = 1           # motif width of the null model
        logP0 = np.sum(  # average log E of null model
            code_book.frequencies*np.log(code_book.frequencies))
        G = None         # model score against null model

        best_phmm_1 = None  # best overall model
        for W in range(window_min, window_max + 1):
            logE = None
            best_phmm_2 = None
            if guess_emissions is None:
                emissions = []
                priors = []
                for i in util.random_pick(range(n - W - 1), t_seeds):
                    f0 = util.guess_emissions(code_book, X[i:i + W])
                    emissions.append(f0)
                    priors.append(1.0 + 1e-3*f0[0, :])
            else:
                emissions, priors = guess_emissions(code_book, W, X, t_seeds)

            for b_out in np.linspace(0, 1, steps + 1, endpoint=False)[1:]:
                for d in np.linspace(0, 1, steps + 1, endpoint=False)[1:]:
                    for s in range(n_seeds):
                        # pick a random subsequence
                        f0 = emissions.pop()
                        t0 = np.random.rand(6)
                        t0[1] = b_out
                        t0[2] = d
                        t0[3:] /= t0[2:].sum()
                        eps = priors.pop()
                        p0 = np.repeat(1.0/(2*W), 2*W)

                        hmm = cls(f=f0, t=t0, p0=p0, eps=eps)
                        hmm.fit_em(X, precision, max_iter)

                        if logE is None or hmm.logE > logE:
                            logE = hmm.logE
                            best_phmm_2 = hmm
                    log.logger.info(
                        'ProfileHMM.fit b_out = {0:.2e} d = {1:.2e} logE = {2:.2e}'.format(b_out, d, logE))
            G2 = util.model_score(n*logP0, logE, (W - W0)*(A - 1))
            log.logger.info(
                'ProfileHMM.fit W = {0} E1 = {1} G = {2}'.format(W, logE, G2))

            if G is None or G2 < G:
                G = G2
                best_phmm_1 = best_phmm_2

            if G == 0:
                log.logger.info('Switched null model (W={0})'.format(W))
                G = 1.0
                logP0 = logE/n
                W0 = float(W)

        best_phmm_1.code_book = code_book
        return best_phmm_1

    def extract(self, X, min_score=-0.7):
        if self.code_book is not None:
            X = np.array(map(self.code_book.code, X))

        Z, logP = self.viterbi(X)
        i_start = None
        i_end = None
        z_end = None

        def valid_motif():
            return (self.score(X[i_start:i_end], Z[i_start:i_end])/
                    float(i_end - i_start + 1)) >= min_score

        for i, z in enumerate(Z):
            if z >= self.W:
                if i_start is None:
                    i_start = i
                    count = 0
                else:
                    if z <= z_end:
                        if valid_motif():
                            yield (i_start, i_end), Z[i_start:i_end]
                        i_start = i
                        count = 0
                i_end = i
                z_end = z
                count += 1
        if i_start is not None and valid_motif():
            yield (i_start, i_end), Z[i_start:i_end]
