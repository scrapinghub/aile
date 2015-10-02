import random

import numpy as np
import scipy.optimize as opt
import scrapely.htmlpage as hp

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
        self.logPZ = np.log(np.where(self._pZ>0, self._pZ, 1e-100))
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
            X[i] = self.dist_e[Z[i  ]].sample()
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


    def viterbi(self, X):
        n = len(X)

        delta = np.zeros((self.S, n))
        psi = np.zeros((self.S, n), dtype=int)

        # log P(Z1, X1)
        delta[:, 0] = self.logPZ + self.logPE[:, X[0]]
        for i in xrange(1, n):
            m = self.logPT.T + delta[:, i - 1]
            psi[: ,i] = np.argmax(m, axis=1)
            # log P(Z1, .., Zi=j, X1, ..., Xi)
            delta[:, i] = m[np.arange(self.S), psi[:, i]] + self.logPE[:, X[i]]

        z = np.zeros((n,), dtype=int)
        z[n - 1] = np.argmax(delta[:, n - 1])
        logP = delta[z[n - 1], n - 1]
        for i in xrange(n - 1, 0, -1):
            z[i - 1] = psi[z[i], i]

        return z, logP


class ProfileHMM(FixedHMM):
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

        # Initialize FixedHMM
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
        pT[0,      0] = b_out
        pT[0, self.W] = 1 - b_out
        for j in xrange(1, self.W):
            pT[j, j         ] = b_in     # b[j] -> b[j]
            pT[j, self.W + j] = 1 - b_in # b[j] -> m[j]
        for j in xrange(self.W):
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
        X = sequence
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
            for j in xrange(1, self.W):
                b1 += c[j,          j] # stay at background
                b2 += c[j, self.W + j] # switch from background to motif
            b_in = b1/(b1 + b2)
            b_out = c[0, 0]/(c[0,0] + c[0, self.W])

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
                r = -k0*util.safe_log(mb) - k1.dot(util.safe_log(s))
                return r

            prange = (1e-6, 1.0 - 1e-6)
            res = opt.fmin_slsqp(
                func        = g,
                x0          = self.t[2:],
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
            self.t = np.array([b_in, b_out, d, mb, m, md])

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
        for W in xrange(window_min, window_max + 1):
            logE = None
            best_phmm_2 = None
            if guess_emissions is None:
                emissions = []
                for i in util.random_pick(xrange(n - W - 1), t_seeds):
                    emissions.append(
                        util.guess_emissions(code_book, X[i:i + W]))
            else:
                emissions = guess_emissions(code_book, W, X, t_seeds)

            for b_out in np.linspace(0, 1, steps + 1, endpoint=False)[1:]:
                for d in np.linspace(0, 1, steps + 1, endpoint=False)[1:]:
                    for s in xrange(n_seeds):
                        # pick a random subsequence
                        f0 = emissions.pop()
                        t0 = np.random.rand(6)
                        t0[1] = b_out
                        t0[2] = d
                        t0[3:] /= t0[2:].sum()
                        eps = 1.0 + 1e-3*f0[0,:]
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


    def extract(self, X, prop1=0.5, prop2=0.25):
        if self.code_book is not None:
            X = np.array(map(self.code_book.code, X))

        Z, logP = self.viterbi(X)
        i_start = None
        z_start = None
        i_end = None
        z_end = None

        def valid_motif(cound):
            return (count >= prop1*float(i_end - i_start) and
                    count >= prop2*float(self.W))

        for i, z in enumerate(Z):
            if z >= self.W:
                if i_start is None:
                    i_start = i
                    z_start = z
                    count = 0
                else:
                    if z <= z_end:
                        if valid_motif(count):
                            yield (i_start, i_end), Z[i_start:i_end]
                        i_start = i
                        z_start = z
                        count = 0
                i_end = i
                z_end = z
                count += 1
        if i_start is not None and valid_motif(count):
            yield (i_start, i_end), Z[i_start:i_end]


def phmm_cmp(W, Z1, Z2):
    return ((Z1 >= W) != (Z2 >= W)).mean()


def demo1():
    phmm_true = ProfileHMM(
        f=np.array([
            [0.2, 0.3, 0.2, 0.3],
            [0.9, 0.1, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2]]),
        t = np.array([
            0.05, 0.9, 0.1, 0.05, 0.85, 0.1])
    )

    X, Z = phmm_true.generate(5000)
    phmm = ProfileHMM.fit(X, 3)

    print "True model 't' parameters", phmm_true.t
    print " Estimated 't' paramaters", phmm.t

    z, logP = phmm.viterbi(X)
    print 'Error finding motifs (% mismatch):', phmm_cmp(phmm.W, Z, z)*100


def html_guess_emissions(code_book, W, X, n=1):
    """Given a sequence X, the code_book used to encode it and the motif
    width W, guess the initial value of the emission matrix"""
    s = code_book.code('/>') # integer for the closing tag symbol
    emissions = []
    # candidates start with a non-closing tag and end with a closing one
    candidates = [i for i in xrange(len(X) - W) if X[i] != s and X[i + W] == s]
    for i in util.random_pick(candidates, n):
        f = util.guess_emissions(code_book, X[i:i+W])
        f[1, s] = 0.0 # zero probability of starting motif with closing tag
        f[W, :] = 0.0 # zero probability of ending motif with non-closing tag
        f[W, s] = 1.0 # probability 1 of ending motif with closing tag
        emissions.append(util.normalized(f))
    return emissions


def demo2():
    page = hp.url_to_page('https://news.ycombinator.com/')
    tags_1 = [fragment for fragment in page.parsed_body if isinstance(fragment, hp.HtmlTag)]
    tags_2 = [
        fragment.tag if fragment.tag_type != hp.HtmlTagType.CLOSE_TAG
        else '/>' for fragment in tags_1
    ]

    cb = util.CodeBook(tags_2)
    X = np.array(tags_2)
    phmm = ProfileHMM.fit(X, 42, guess_emissions=html_guess_emissions)
    for (i, j), Z in phmm.extract(X):
        print 80*'#'
        print Z
        print 80*'-'
        print page.body[tags_1[i].start:tags_1[j].end]
    return phmm


if __name__ == '__main__':
    phmm = demo2()
