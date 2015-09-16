import collections
import logging
import random

import numpy as np
import scipy.optimize as opt
import scrapely.htmlpage as hp


class Sequence(object):
    def __init__(self, iterable, start_at=lambda x: True):
        """Iterate over iterable using windows of the given size.

        For example:

        s = Sequence(range(8))
        for window in s.roll(3):
            print window

        Outputs:

        [0, 1, 2]
        [1, 2, 3]
        [2, 3, 4]
        [3, 4, 5]
        [4, 5, 6]
        [5, 6, 7]
        """
        self._iterable = iterable
        self.start_at = start_at

    def __iter__(self):
        return iter(self._iterable)

    def roll(self, window_size):
        window = collections.deque([], maxlen=window_size)
        it = enumerate(iter(self._iterable))
        # Make sure to start at a valid element
        for i, element in it:
            if self.start_at(element):
                window.append(element)
                break
        # Initial fill of the window
        for i, element in it:
            window.append(element)
            if len(window) == window_size:
                break
        if len(window) > 0:
            yield (i - window_size + 1, tuple(window))
        else:
            return
        # Return windows at valid starting positions
        for i, element in it:
            window.popleft()
            window.append(element)
            if self.start_at(window[0]):
                yield (i - window_size + 1, tuple(window))


class HTMLSequence(Sequence):
    """A sequence over HTML tags that always starts the window at an opening
    tag."""
    def __init__(self, html):
        self.tags = filter(
            lambda fragment: isinstance(fragment, hp.HtmlTag),
            html.parsed_body)
        super(HTMLSequence, self).__init__(
            [(x.tag, x.tag_type) for x in self.tags],
            start_at=lambda x: x[1] == hp.HtmlTagType.OPEN_TAG
        )


class CodeBook(object):
    """
    Map symbols to integer, allowing to code and decode sequences using
    this mapping
    """
    def __init__(self, sequence):
        self.counter = collections.Counter(sequence)
        self.letters = sorted(set(self.counter))
        self._index = {
            letter: index
            for index, letter in enumerate(self.letters)
        }
        self.total_count = sum(self.counter.values())
        self.frequencies = np.array(
            [self.counter[letter] for letter in self.letters],
            dtype=float
        )
        self.frequencies /= float(self.total_count)

    def __len__(self):
        """Number of symbols"""
        return len(self.letters)

    def code(self, letter):
        """Convert letter to integer.

        Note: if letter is not in the code book then insert None
        inside the output sequence.
        """
        return self._index.get(letter, None)

    def decode(self, index):
        """Perform the inverse operation of code.

        Note: as in coding, if some symbol can't be found then insert None.
        """
        N = len(self.letters)
        if index >= 0 and index < N:
            return self.letters[index]
        else:
            return None


def normalized(P):
    """Return an stochastic matix (all rows sum to 1), proportional to P"""
    return (P.T/P.sum(axis=1)).T


class MEME(object):
    """Implementation of the MEME algorithm (not complete).

    MEME tries to find repeating patterns inside an stream of symbols. It was
    originally conceived to find patterns inside long molecules sequences
    (think DNA).

    The following papers contain a description of the algorithm:

    - Unsupervised learning of multiple motifs in biopolymers using
      expectation maximization.
      Bailey and Elkan, 1995

    - Fitting a mixture model by expectation maximization to discover
      motifs in bipolymers.
      Bailey and Elkan, 1994

    - The value of prior knowledge in discovering motifs with MEME.
      Bailey and Elkan, 1995

    The key phrase to search more papers is "motif discovery".

    TODO:
        - Auto-selecting motif length.
    """
    def __init__(self, window=8, logger='default'):
        self.window = window
        if logger is 'default':
            logging.basicConfig()
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger


    def _roll(self, sequence):
        return enumerate(
            np.array([self.code_book.code(x) for x in window])
            for _, window in sequence.roll(self.window))


    def _start(self, X, gamma=0.1):
        A = len(self.code_book)
        W = len(X)
        p = np.log(1.0/float(A))
        m = opt.newton(
            lambda m: (1.0 - m)*np.log((1.0 - m)/(A - 1)) - p + m*np.log(m) + gamma*p,
            0.5
        )
        # Position Frequency Matrix
        # columns    : one for each code_book letter
        # row 0      : background frequency
        # row 1 .. W : motif frequency
        f = np.zeros((W + 1, A))
        f[0, :] = self.code_book.frequencies
        for j in xrange(W):
            f[j + 1, :   ] = (1.0 - m)/(A - 1)
            f[j + 1, X[j]] = m
        return f

    @staticmethod
    def _score(X, f):
        pM = 1.0 # probability of X taken from motif
        pB = 1.0 # probability of X taken from background
        for j, k in enumerate(X):
            pM *= f[j + 1, k]
            pB *= f[0    , k]
        return pM, pB

    def score(self, X):
        """Return the motif and background probabilities of seeing the
        sub-sequence X"""
        return np.array([
            MEME._score(map(self.code_book.code, X),
                        self.position_frequency_matrix[k,:,:])
            for k in xrange(self.n_motifs)])

    def _fit_1(self, sequence, V, f, m, eps, max_iter, relax):
        """Run the Expectation-Maximization algorithm"""
        W = self.window                        # size of window
        n = len(V)                             # number of windows
        it = 0                                 # current iteration
        E1 = None                              # previous expected log-likelihood
        while True:
            Z = np.zeros((n,)) # E[X[i] is motif]
            E2 = 0.0           # current expected log-likelihood
            for i,  X in self._roll(sequence):
                # pM: P(X|X is motif     )
                # PB: P(X|X is background)
                pM, pB = MEME._score(X, f)
                Z[i] = (pM*m/(pM*m + pB*(1 - m)))
                # TODO: correct this formula to take into account:
                #     a. Beta prior
                #     b. Erasure coefficients
                E2 += Z[i]*np.log(pM*m) + (1 - Z[i])*np.log(pB*(1 - m))

            # Correct for overlapping, enforce Z[i:(i+W)] <= 1 for i=1...n
            # MEME hack to account for dependence between sub-sequences.
            for i in xrange(len(Z)):
                s = Z[i:i+W].sum()
                if s > 1.0:
                    Z[i:i+W] /= s

            q0 = 0.0
            q1 = 0.0
            c = np.zeros(f.shape, dtype=float)
            for i,  X in self._roll(sequence):
                Z0 = 1.0 - Z[i]
                Z1 = Z[i]*V[i]
                for j, k in enumerate(X):
                    c[0    , k] += Z0
                    c[j + 1, k] += Z1
                q0 += Z0
                q1 += Z1
            m = q1/(q0 + q1)
            f = (1.0 - relax)*normalized(c) + relax
            if E1:
                err = (E1 - E2)/E1
                if err < eps:
                    break

            E1 = E2
            it += 1
            if it > max_iter:
                if self.logger:
                    self.logger.warning(
                        'MEME.fit: max iterations reached without convergence (err={0})'.format(err))
                break

        return f, m, E2, Z

    def fit(self, sequence, n_motifs=1, gamma=0.1, alpha=0.5, eps=1e-5, max_iter=100, relax=1e-2):
        self.code_book = CodeBook(sequence)
        self.n_motifs = n_motifs

        A = len(self.code_book)                        # alphabet size
        W = self.window                                # window size
        n = sum(1 for _ in sequence.roll(self.window)) # sequence length

        self.em_objective = np.zeros((n_motifs,))
        self.position_frequency_matrix = np.zeros((n_motifs, W+1, A))
        self.motif_probability = np.zeros((n_motifs,))

        U = np.ones((n,))
        V = np.ones((n,))
        for k in xrange(n_motifs):
            E = None
            m0 = 1.0/n
            while m0 < 1.0/W:
                Q = max(1, np.log(1.0 - alpha)/np.log(1.0 - m0))
                # Select subsequences as seeds
                r = set(np.random.randint(0, n, Q))
                seeds = []
                for i, X in self._roll(sequence):
                    if i in r:
                        seeds.append(X)
                for s, seed in enumerate(seeds):
                    f0 = self._start(seed, gamma)
                    f1, m1, E1, Z1 = self._fit_1(sequence, V, f0, m0, eps, max_iter, relax)
                    if not E or E1 > E:
                        f, m, E, Z = f1, m1, E1, Z1
                    if self.logger:
                        self.logger.info(
                            'MEME.fit: pass={0}, m={1:.2e}, seed={2}, E[log(P)]={3:.2e}, best={4:.2e}'.format(
                                k, m0, s, E1, E))
                m0 *= 2.0

            # Save motif
            self.em_objective[k] = E
            self.position_frequency_matrix[k,:,:] = f
            self.motif_probability[k] = m

            # Update erasure
            for i in xrange(n):
                U[i] *= 1.0 - V[i]*Z[max(0, i - W + 1):(i + 1)].max()
            for i in xrange(n):
                V[i] = U[i:(i+W)].min()

        self._motif_threshold = np.log((1.0 - self.motif_probability)/self.motif_probability)

    def find_motif(self, sequence):
        for i, X in sequence.roll(self.window):
            score = self.score(X)
            Z = np.log(score[:,0]/score[:,1])
            for k in xrange(self.n_motifs):
                if Z[k] > self._motif_threshold[k]:
                    yield MotifMatch(
                        index=i,
                        group=k,
                        score=Z[k],
                        motif=X
                    )


class MotifMatch(object):
    def __init__(self, index, group, score, motif):
        """The results of MEME.find_motif.

        Fields:
            - index: position of the motif inside the sequence
            - group: integer identifying the matching motif group
            - score: the higher the better
            - motif: the matched sub-sequence
        """
        self.index = index
        self.group = group
        self.score = score
        self.motif = motif

    def __str__(self):
        return 'index={0}, group={1}, score={2:.2e}, motif={3}'.format(
            self.index, self.group, self.score, self.motif)

    def __repr__(self):
        return self.__str__()


def demo1():
    a = ['A', 'B', 'C', 'D']
    s = ''
    t = set()
    k = 0
    for i in xrange(60):
        r = random.randint(0, 10)
        if r <= 8:
            s += a[r % 4]
            k += 1
        else:
            t.add(k)
            if r == 9:
                s += 'ABC'
            else:
                s += 'DAD'
            k += 3

    m = MEME(3)
    m.fit(Sequence(s), 2)
    print s
    for match in m.find_motif(Sequence(s)):
        if match.index in t:
            print (match.index - 1)*' ' + '>' + ''.join(match.motif) + '<'
            t.remove(match.index)
        else:
            print (match.index - 1)*' ' + 'x' + ''.join(match.motif) + 'x'
    if t:
        print 'Missing:'
        print s
        for i in t:
            print i*' ' + s[i:i+3]


def demo2():
    p = hp.url_to_page('http://www.elmundo.es')
    s = HTMLSequence(p)
    m = MEME(4)
    m.fit(s, 1)
    for match in m.find_motif(s):
        print 80*'-'
        print 'SCORE =', match.score
        print 'GROUP =', match.group
        print 'MOTIF =', match.motif
        print 'HTML'
        print '****'
        print p.body[s.tags[match.index].start:s.tags[match.index+3].end]


if __name__ == '__main__':
    demo1()
