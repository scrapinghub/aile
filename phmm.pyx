import math
import numpy as np
import scipy.optimize as opt
from joblib import Parallel, delayed

import util
import hmm

cimport numpy as np
cimport cython


def fit_best_seed(X, seeds, warmup=3, precision=1e-5, max_iter=200):
    return ProfileHMM.fit_best_seed(X, seeds, warmup, precision, max_iter)


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
        self.W = f.shape[0] - 1 # motif length

        self._f = f
        self._t = t

        self.eps = eps if eps is not None else np.repeat(1.0, self.A)
        self.p0 = p0 if p0 is not None else np.repeat(1.0/self.S, self.S)

        self.code_book = None

        # Initialize hmm.FixedHMM
        super(ProfileHMM, self).__init__(self.p0, self.calc_pE(f), self.calc_pT(t))

    def calc_pE(self, f):
        pE = np.zeros((2*self.W, f.shape[1]))
        pE[:self.W, :] = f[ 0, :] # copy W times the background emissions
        pE[self.W:, :] = f[1:, :] # copy as is the motif probabilities
        return pE

    def calc_pT(self, t):
        b_in, b_out, d, mb, m, md = t
        pT = np.zeros((2*self.W, 2*self.W))
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
        self.set_pT(self.calc_pT(value))

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self._f = value
        self.set_pE(self.calc_pE(value))

    def fit_em_1(self, np.ndarray[np.int_t, ndim=1] sequence):
        """Run a single iteration of the EM method"""
        fb = self.forward_backward(sequence)
        # Take into account the priors on the parameters
        fb.logP += np.sum((self.eps - 1)*util.safe_log(self.f[1:,:]))

        cdef unsigned int W = self.W
        cdef np.ndarray[np.double_t, ndim=2] gamma = fb.gamma
        cdef np.ndarray[np.double_t, ndim=3] xi = fb.xi

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
        def EM2(np.ndarray[np.double_t, ndim=1] tM,
                np.ndarray[np.double_t, ndim=2] c):
            cdef double d  = tM[0]
            cdef double mb = tM[1]
            cdef double m  = tM[2]
            cdef double md = tM[3]

            # numerical derivation can make these magnitudes get near zero
            if d  <= 1e-100: d  = 1e-100
            if mb <= 1e-100: mb = 1e-100
            if m  <= 1e-100: m  = 1e-100
            if md <= 1e-100: md = 1e-100

            cdef unsigned int W = c.shape[0]/2

            cdef np.ndarray[np.double_t, ndim=1] F = (
                np.log(d)*np.arange(W) +
                np.log(md) +
                np.log(1 - d) -
                np.log(1 - d**W)
            )
            F[W - 1] = np.logaddexp(F[W - 1], np.log(m))

            r = 0
            for j in range(W):
                r -= c[W + j, (j + 1) % W]*np.log(mb)
                for k in range(W):
                    r -= c[W + j, W + k]*F[(k - j - 2) % W]

            return r

        # Equality constraint for tM parameters
        def gm(np.ndarray[np.double_t, ndim=1] tM,
               np.ndarray[np.double_t, ndim=2] c):
            cdef double mb = tM[1]
            cdef double m  = tM[2]
            cdef double md = tM[3]

            return (mb + m + md) - 1.0

        d, mb, m, md = opt.fmin_slsqp(
            func        = EM2,
            x0          = self.t[2:].copy(),
            bounds      = 4*[(1e-3, 1.0 - 1e-3)],
            eqcons      = [gm],
            iprint      = 0,
            args        = (c/n, ),
            epsilon     = 1e-6,
            acc         = 1e-9
        )

        self.f = f
        self.t = np.array([b_in, b_out, d, mb, m, md])

        return fb.logP

    def fit_em_n(self, sequence, n=1):
        """Iterate the EM algorithm 'n' times"""
        for n in range(n):
            logP = self.fit_em_1(sequence)
        return logP

    def fit_em(self,
               np.ndarray[np.int_t, ndim=1] sequence,
               double precision=1e-3,
               unsigned int max_iter=100):
        logP0 = None
        it = 0
        while True:
            logP1 = self.fit_em_1(sequence)
            # Check convergence
            if logP0:
                err = (logP0 - logP1)/logP0
                if err < 0:
                    # should never happen
                    self.logger.warning(
                        'ProfileHMM.fit_em: log(P) decreased {0}({1:3f}%)'.format(logP0, err*100.0))
                    break
                if err < precision:
                    break
            logP0 = logP1
            it += 1
            if it > max_iter:
                # TODO
                break
        return logP1

    @staticmethod
    def seeds(X, code_book, W, rB, rD, guess_emissions=None):
        """Given a sequence X, which has been coded using code_book,
        and a range of b_out (rB) and d (rD) return a set of initial
        parameters"""

        n_seeds = len(rB)*len(rD)
        if guess_emissions is None:
            f0 = [util.guess_emissions(code_book, X[i:i + W])
                  for i in util.random_pick(range(len(X) - W - 1), n_seeds)]
            eps = [1.0 + 1e-3*f[0, :] for f in f0]
        else:
            f0, eps = guess_emissions(code_book, W, X, n_seeds)

        def gen_random_t0(b_out, d):
            t = np.random.rand(6)
            t[1] = b_out
            t[2] = d
            t[3:] /= t[2:].sum()
            return t
        t0 = [gen_random_t0(b_out, d) for b_out in rB for d in rD]
        p0 = [np.repeat(1.0/(2*W), 2*W) for b_out in rB for d in rD]

        return zip(f0, t0, p0, eps)

    @classmethod
    def fit_best_seed(cls, X, seeds, warmup=3, precision=1e-5, max_iter=200):
        best_phmm = None
        best_logP = None
        for (f0, t0, p0, eps) in seeds:
            phmm = cls(f=f0, t=t0, p0=p0, eps=eps)
            logP = phmm.fit_em_n(X, warmup)
            if best_logP is None or logP > best_logP:
                best_logP = logP
                best_phmm = phmm
        best_logP = best_phmm.fit_em(X, precision=precision, max_iter=max_iter)
        return best_phmm, best_logP

    @classmethod
    def fit(cls, sequence, window_min, window_max=None,
            gamma=0.1, steps=4, n_seeds=1, precision=1e-5,
            max_iter=200, guess_emissions=None):
        """Fit the model parameters using data.

        - sequence       : the sequence to fit
        - window_min     : minimum motif width to consider
        - window_max     : maximum motif width to consider.
                           If None only one motif width will be considered.
        - gamma          : parameter to pass to the guess_emissions function
        - steps          : number of steps in the grid (b_out,d)
        - n_seeds        : number of seeds per grid element
        - precision      : EM desired precision in logP
        - max_iter       : maximum iterations in EM
        - guess_emissions: a function to compute the initial value of the
                           emissions matrix. Signature:

                               guess_emissions(code_book, W, X)

                           where:
                                code_book: CodeBook used to encode X
                                W        : motif width
                                X        : encoded sequence
        """
        if window_max == None:
            window_max = window_min

        code_book = util.CodeBook(sequence)
        X = np.array(map(code_book.code, sequence))

        rB = rD = np.linspace(0, 1, steps + 1, endpoint=False)[1:]
        rW = range(window_min, window_max + 1)
        best = Parallel(n_jobs=-1)(delayed(fit_best_seed)(
                    X,
                    seeds     = ProfileHMM.seeds(X, code_book, W, rB, rD, guess_emissions),
                    warmup    = 3,
                    precision = precision,
                    max_iter  = max_iter
                ) for W in rW)
        # Null model
        A = len(code_book)
        logP0 = len(sequence)*np.sum(
            code_book.frequencies*np.log(code_book.frequencies))
        models = [(None, logP0, A - 1)]
        for (phmm, logP), W in zip(best, rW):
            models.append((phmm, logP, W*(A - 1)))
        best_phmm = util.model_select(models)

        #-----------------------------------------------------------------------
        # WARNING
        #-----------------------------------------------------------------------
        # When running this in parallel it seems that FixedHMM attributes
        # get destroyed. Maybe some weird interaction between cython and joblib?
        # It looks like some pickling error or who knows what. As a hack we
        # rebuild the model,  it seems that parameters f and t arrive allright
        #-----------------------------------------------------------------------
        res = cls(f=best_phmm.f, t=best_phmm.t, eps=best_phmm.eps, p0=best_phmm.p0)
        res.code_book = code_book
        return res

    def extract(self, X, min_score=-2.0):
        if self.code_book is not None:
            X = np.array(map(self.code_book.code, X))

        Z, logP = self.viterbi(X)
        i_start = None
        i_end = None
        z_end = None

        def valid_motif():
            if min_score is None:
                return True
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
                            yield (i_start, i_end+1), Z[i_start:i_end+1]
                        i_start = i
                        count = 0
                i_end = i
                z_end = z
                count += 1
        if i_start is not None and valid_motif():
            yield (i_start, i_end+1), Z[i_start:i_end+1]
