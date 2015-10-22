import time
import math
import numpy as np
import scipy.optimize as opt
from joblib import Parallel, delayed

import util
import hmm

cimport numpy as np
cimport cython


# Module-level function wrappers to allow Parallel execution
############################################################
def fit_seed(X, seed, warmup):
    f0, t0, p0, eps = seed
    phmm = ProfileHMM(f=f0, t=t0, p0=p0, eps=eps)
    logP = phmm.fit_em_n(X, warmup)
    return (logP, (phmm.f, phmm.t, phmm.p0, phmm.eps))


def fit_em(X, params, precision, max_iter):
    phmm = ProfileHMM(f=params[0], t=params[1], p0=params[2], eps=params[3])
    logP = phmm.fit_em(X, precision=precision, max_iter=max_iter)
    return phmm, logP


# Moved out of class, to declare as cdef to improve performance
@cython.boundscheck(False)
cdef forward_backward(np.ndarray[np.int_t, ndim=1] X,
                      np.ndarray[np.double_t, ndim=1] pZ,
                      np.ndarray[np.double_t, ndim=2] pE,
                      np.ndarray[np.double_t, ndim=2] pT,
                      unsigned int w = 4):
    """Run the forward-backwards algorithm using data X.

    Compute the following magnitudes using the current parameters:

        - P(X[:i+1]          , Z[i]=j         )      alpha [j,    i]
        - P(X[i:  ]                   | Z[i]=j)      beta  [j,    i]
        - P(         Z[i-1]=j, Z[i]=k | X     )      xi    [j, k, i]
        - P(         Z[i  ]=j         | X     )      gamma [j,    i]

        - sum E   [log P(X,Z)]                       logP
              Z|X
    """

    cdef unsigned int S = pE.shape[0]
    cdef unsigned int A = pE.shape[1]
    cdef int W = S/2

    cdef unsigned int n = len(X) # sequence length
    cdef unsigned int i          # index from 0 to n - 1
    cdef double a, b             # accumulators
    cdef int s, t, u, v          # index from 0 to S - 1

    cdef np.ndarray[np.double_t, ndim=2] pS = np.zeros((S, S))
    for u in range(W):
        for v in range(W):
            pS[    u,     v] = pT[    u,     v]
            pS[    u, W + v] = pT[    u, W + v]
            pS[W + u,     v] = pT[W + u,     v]
        for v in range(w):
            t = W + (u + v + 1) % W
            pS[W + u,  t] = pT[W + u, t]
    for s in range(S):
        a = 0
        for t in range(S):
            a += pS[s, t]
        for t in range(S):
            pS[s, t] /= a

    # to avoid numerical precision errors we multiply alpha and beta
    # by this number
    cdef np.ndarray[np.double_t, ndim=1] scale = np.zeros((n, ))

    # alpha[j, i] = P(X[:i+1], Z[i]=j)
    #             = sum_t{pT[t, s]alpha[t, i -1]}pE[s, X[i]]
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
        for u in range(W):
            b = 0
            s = u
            t = u
            b += pS[t, s]*alpha[t, i-1]
            t = W + (u - 1) % W
            b += pS[t, s]*alpha[t, i-1]
            b *= pE[s, X[i]]
            alpha[s, i] = b
            a += b

            b = 0
            s = W + u
            t = u
            b += pS[t, s]*alpha[t, i-1]
            for v in range(w):
                t = W + (u - v - 1) % W
                b += pS[t, s]*alpha[t, i-1]
            b *= pE[s, X[i]]
            alpha[s, i] = b
            a += b

        if a > 0:
            a = 1.0/a
        for s in range(S):
            alpha[s, i] *= a
        scale[i] = a

    # beta[s, i] = P(X[i:] | Z[i]=s)
    #            = sum_t{pT[s, t]beta[t, i +1]}pE[s, X[i]]
    cdef np.ndarray[np.double_t, ndim=2] beta = np.zeros((S, n))
    for s in range(S):
        beta[s, n-1] = pE[s, X[n-1]]*scale[n-1]
    for i in range(n - 1, 0, -1):
        a = scale[i - 1]
        for u in range(W):
            b = 0
            s = u
            t = u
            b += pS[s, t]*beta[t, i]
            t = W + u
            b += pS[s, t]*beta[t, i]
            beta[s, i - 1] = b*a*pE[s, X[i - 1]]

            b = 0
            s = W + u
            t = (u + 1) % W
            b += pS[s, t]*beta[t, i]
            for v in range(w):
                t = W + (u + v + 1) % W
                b += pS[s, t]*beta[t, i]
            beta[s, i - 1] = b*a*pE[s, X[i - 1]]

    # xi[j, k, i] = P(Z[i-1]=j, Z[i]=k | X)
    cdef np.ndarray[np.double_t, ndim=2] xi_i = np.zeros((S, S))
    cdef np.ndarray[np.double_t, ndim=2] xi_c = np.zeros((S, S))
    cdef np.ndarray[np.double_t, ndim=2] gm_c = np.zeros((S, A))

    for i in range(1, n):
        a = 0
        for u in range(W):
            s = u
            t = u
            xi_i[s, t] = b = alpha[s, i - 1]*beta[t, i]*pS[s, t]
            a += b
            t = W + u
            xi_i[s, t] = b = alpha[s, i - 1]*beta[t, i]*pS[s, t]
            a += b
            s = W + u
            t = (u + 1) % W
            xi_i[s, t] = b = alpha[s, i - 1]*beta[t, i]*pS[s, t]
            a += b
            for v in range(w):
                t = W + (u + v + 1) % W
                xi_i[s, t] = b = alpha[s, i - 1]*beta[t, i]*pS[s, t]
                a += b
        for u in range(W):
            s = u
            t = u
            xi_i[s, t] /= a
            t = W + u
            xi_i[s, t] /= a
            s = W + u
            t = (u + 1) % W
            xi_i[s, t] /= a
            for v in range(w):
                t = W + (u + v + 1) % W
                xi_i[s, t] /= a
        if i==1:
            for u in range(W):
                s = u
                t = u
                gm_c[s, X[0]] += xi_i[s, t]
                t = W + u
                gm_c[s, X[0]] += xi_i[s, t]
                s = W + u
                t = (u + 1) % W
                gm_c[s, X[0]] += xi_i[s, t]
                for v in range(w):
                    t = W + (u + v + 1) % W
                    gm_c[s, X[0]] += xi_i[s, t]
        else:
            for u in range(W):
                s = u
                t = u
                xi_c[s, t   ] += xi_i[s, t]
                gm_c[s, X[i]] += xi_i[t, s]
                t = W + u
                xi_c[s, t   ] += xi_i[s, t]
                gm_c[s, X[i]] += xi_i[t, s]
                s = W + u
                t = (u + 1) % W
                xi_c[s, t   ] += xi_i[s, t]
                gm_c[s, X[i]] += xi_i[t, s]
                for v in range(w):
                    t = W + (u + v + 1) % W
                    xi_c[s, t   ] += xi_i[s, t]
                for v in range(w):
                    t = W + (u - v - 1) % W
                    gm_c[s, X[i]] += xi_i[t, s]

    res = hmm.FBResult()
    res.scale = scale
    res.alpha = alpha
    res.beta = beta
    res.xi = xi_c
    res.gamma = gm_c
    res.logP = -np.log(scale).sum()
    return res


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
        self.p0 = p0 if p0 is not None else np.repeat(1.0/(2*self.W), 2*self.W)
        # Initialize hmm.FixedHMM
        super(ProfileHMM, self).__init__(self.p0, self.calc_pE(f), self.calc_pT(t))

        self._f = f
        self._t = t

        self.eps = eps if eps is not None else np.repeat(1.0, self.A)

    def calc_pE(self, f):
        """Compute the emission matrix given the parameters 'f'"""
        pE = np.zeros((2*self.W, f.shape[1]))
        pE[:self.W, :] = f[ 0, :] # copy W times the background emissions
        pE[self.W:, :] = f[1:, :] # copy as is the motif probabilities
        return pE

    def calc_pT(self, t):
        """Compute the transition matrix given the parameters 't'"""
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

    # Wrapper around the cdef function
    def forward_backward(self,
                         np.ndarray[np.int_t, ndim=1] X,
                         unsigned int w=3):
        return forward_backward(X, self.pZ, self.pE, self.pT, w)

    def fit_em_1(self, np.ndarray[np.int_t, ndim=1] sequence):
        """Run a single iteration of the EM method.

        Improves the current f and t parameters and returns logP (the logarithm
        of the probability of the parameters given the data)
        """
        fb = self.forward_backward(sequence)
        # Take into account the priors on the parameters (Dirichlet prior)
        fb.logP += np.sum((self.eps - 1)*util.safe_log(self.f[1:,:]))

        cdef unsigned int W = self.W
        cdef unsigned int A = self.A
        cdef np.ndarray[np.double_t, ndim=2] gamma = fb.gamma
        cdef np.ndarray[np.double_t, ndim=2] xi = fb.xi

        cdef np.ndarray[np.int_t, ndim=1] X = sequence
        cdef unsigned int n = len(X)

        cdef unsigned int a, i, j, k, l

        cdef np.ndarray[np.double_t, ndim=2] f = np.zeros(self.f.shape)
        for a in range(A):
            for s in range(W):
                f[    0, a] += gamma[    s, a]
                f[1 + s, a] += gamma[W + s, a]
        f[0,  :] /= f[0, :].sum()
        f[1:, :] = util.normalized(f[1:, :] + self.eps - 1.0)

        # count state transitions
        cdef double b1 = 0.0
        cdef double b2 = 0.0
        for j in range(1, W):
            b1 += xi[j,     j] # stay at background
            b2 += xi[j, W + j] # switch from background to motif
        cdef double b_in = b1/(b1 + b2)
        cdef double b_out = xi[0, 0]/(xi[0,0] + xi[0, W])

        cdef np.ndarray[np.double_t, ndim=2] pT = self.pT

        # objective function for parameters
        def EM2(np.ndarray[np.double_t, ndim=1] tM,
                np.ndarray[np.double_t, ndim=2] xi):
            cdef double d  = tM[0]
            cdef double mb = tM[1]
            cdef double m  = tM[2]
            cdef double md = tM[3]

            # numerical derivation can make these magnitudes get near zero
            if d  <= 1e-100: d  = 1e-100
            if mb <= 1e-100: mb = 1e-100
            if m  <= 1e-100: m  = 1e-100
            if md <= 1e-100: md = 1e-100

            cdef unsigned int W = xi.shape[0]/2

            cdef np.ndarray[np.double_t, ndim=1] F = (
                np.log(d)*np.arange(W) +
                np.log(md) +
                np.log(1 - d) -
                np.log(1 - d**W)
            )
            F[W - 1] = np.logaddexp(F[W - 1], np.log(m))

            r = 0
            for j in range(W):
                r -= xi[W + j, (j + 1) % W]*np.log(mb)
                for k in range(W):
                    r -= xi[W + j, W + k]*F[(k - j - 2) % W]

            return r

        # Equality constraint for tM parameters
        def gm(np.ndarray[np.double_t, ndim=1] tM,
               np.ndarray[np.double_t, ndim=2] xi):
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
            args        = (xi/n, ),
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
        """Iterate the EM algorithm until we get the desired precision"""
        logP0 = None
        it = 0
        while True:
            logP1 = self.fit_em_1(sequence)
            # Check convergence
            if logP0:
                err = (logP0 - logP1)/logP0
                if err < 0:
                    # TODO
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
    def frequencies(X, W, A):
        f = np.bincount(X, minlength=A)
        f = f.astype(float)
        return f/f.sum()

    @staticmethod
    def seeds(X, W, A, rB, rD, guess_emissions=None, gamma=0.1):
        """Given a sequence X and a range of b_out (rB) and d (rD) return a
        set of initial parameters"""
        n_seeds = len(rB)*len(rD)
        if guess_emissions is None:
            freqs = ProfileHMM.frequencies(X, W, A)
            f0 = [util.guess_emissions(freqs, X[i:i + W], gamma)
                  for i in util.random_pick(range(len(X) - W - 1), n_seeds)]
            eps = [1.0 + 1e-3*f[0, :] for f in f0]
        else:
            f0, eps = guess_emissions(W, n_seeds)

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
    def fit_seed(cls, X, seeds, warmup=3, precision=1e-5, max_iter=200):
        """Fit the seeds during 'warmup' iterations of the EM algorithm and
        return the most promising seed after these iterations"""
        improved = Parallel(n_jobs=-1)(
            delayed(fit_seed)(X, seed, warmup) for seed in seeds)
        best_seed = None
        best_logP = None
        for logP, seed in improved:
            if best_logP is None or logP > best_logP:
                best_logP = logP
                best_seed = seed
        return best_seed

    @classmethod
    def fit(cls, X, widths, gamma=0.1, steps=4,
            precision=1e-5, max_iter=200, guess_emissions=None):
        """Fit the model parameters using data.

        - X              : the sequence to fit
        - widths         : motif width to consider
        - gamma          : parameter to pass to the guess_emissions function
        - steps          : number of steps in the grid (b_out,d)
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
        A = max(X) + 2
        rB = rD = np.linspace(0, 1, steps + 1, endpoint=False)[1:]
        best_seeds = [cls.fit_seed(
                        X,
                        seeds     = ProfileHMM.seeds(X, W, A, rB, rD, guess_emissions, gamma),
                        warmup    = 3,
                        precision = precision,
                        max_iter  = max_iter
                    ) for W in widths]
        best = Parallel(n_jobs=-1)(
            delayed(fit_em)(X, seed, precision, max_iter) for seed in best_seeds)

        #-----------------------------------------------------------------------
        # WARNING
        #-----------------------------------------------------------------------
        # When running this in parallel it seems that FixedHMM attributes
        # get destroyed. Maybe some weird interaction between cython and joblib?
        # It looks like some pickling error or who knows what. As a hack we
        # rebuild the model,  it seems that parameters f and t arrive allright
        #-----------------------------------------------------------------------
        return [(cls(f=phmm.f, t=phmm.t, eps=phmm.eps, p0=phmm.p0), logP)
                for phmm, logP in best]
