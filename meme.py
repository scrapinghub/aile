import collections
import itertools
import logging
import random

import numpy as np
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
        it = itertools.dropwhile(
            lambda x: not self.start_at(x), iter(self._iterable))
        window = collections.deque(
            itertools.islice(it, window_size) , maxlen=window_size)
        if len(window) > 0:
            yield tuple(window)
        else:
            return
        for element in it:
            window.popleft()
            window.append(element)
            if self.start_at(window[0]):
                yield tuple(window)


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

    def decode(self, sequence):
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

    - The value of prior knowledge in discovering motifs with MEME.
      Bailey and Elkan, 1995

    The key phrase to search more papers is "motif discovery".

    TODO:
        - EM starting points are currently selected randomly, instead of using
          MEME heuristic.
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

        def roll(sequence):
            return enumerate(
                np.array([self.code_book.code(x) for x in window])
                for window in sequence.roll(self.window))

        W = self.window                        # size of window
        n = self.code_book.total_count - W + 1 # number of windows
        it = 0                                 # current iteration
        E1 = None                              # previous expected log-likelihood
        while True:
            Z = np.zeros((n,)) # E[X[i] is motif]
            E2 = 0.0           # current expected log-likelihood
            for i, X in roll(sequence):
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
            for i, X in roll(sequence):
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

    def fit(self, sequence, n_motifs=1, n_seeds=10, eps=1e-5, max_iter=100, relax=1e-2):
        self.code_book = CodeBook(sequence)
        self.n_motifs = n_motifs

        A = len(self.code_book)                # alphabet size
        W = self.window                        # window size
        n = self.code_book.total_count - W + 1 # sequence length

        self.em_objective = np.zeros((n_motifs,))
        self.position_frequency_matrix = np.zeros((n_motifs, W+1, A))
        self.motif_probability = np.zeros((n_motifs,))

        U = np.ones((n,))
        V = np.ones((n,))
        for k in xrange(n_motifs):
            E = None
            for s in xrange(n_seeds):
                # Position Frequency Matrix
                # columns    : one for each code_book letter
                # row 0      : background frequency
                # row 1 .. W : motif frequency
                f0 = (1.0 - relax)*normalized(np.random.rand(W + 1, A)) + relax
                # A priori probability of motif
                m0 = np.random.random()

                f1, m1, E1, Z1 = self._fit_1(sequence, V, f0, m0, eps, max_iter, relax)

                if not E or E1 > E:
                    f = f1
                    m = m1
                    E = E1
                    Z = Z1

                if self.logger:
                    self.logger.info(
                        'MEME.fit: pass={0}, seed={1}, E[log(P)]={2:.2e}, best={3:.2e}'.format(k, s, E1, E))

            # Update erasure
            for i in xrange(n):
                U[i] *= 1.0 - V[i]*Z[max(0, i - W + 1):(i + 1)].max()
            for i in xrange(n):
                V[i] = U[i:(i+W)].min()

            self.em_objective[k] = E
            self.position_frequency_matrix[k,:,:] = f
            self.motif_probability[k] = m

        self._motif_threshold = np.log((1.0 - self.motif_probability)/self.motif_probability)

    def find_motif(self, sequence):
        for j, X in enumerate(sequence.roll(self.window)):
            score = self.score(X)
            Z = np.log(score[:,0]/score[:,1])
            for k in xrange(self.n_motifs):
                if Z[k] > self._motif_threshold[k]:
                    yield j, k, Z[k], X


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
    for i, j, score, motif in m.find_motif(Sequence(s)):
        if i in t:
            print (i-1)*' ' + '>' + ''.join(motif) + '<'
            t.remove(i)
        else:
            print (i-1)*' ' + 'x' + ''.join(motif) + 'x'
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
    for x in m.find_motif(s):
        print x[3]


if __name__ == '__main__':
    demo1()
