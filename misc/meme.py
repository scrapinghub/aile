import collections
import random
import string
import itertools

import numpy as np
import scrapely.htmlpage as hp
import tabulate

import util


class Sequence(object):
    def __init__(self, iterable):
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

    def __iter__(self):
        return iter(self._iterable)

    def __str__(self):
        return str(self._iterable)

    def __repr__(self):
        return '<Sequence: {0}>'.format(self)

    def roll(self, window_size):
        window = collections.deque([], maxlen=window_size)
        it = enumerate(iter(self._iterable))
        # Initial fill of the window
        for i, element in it:
            window.append(element)
            if len(window) == window_size:
                break
        if len(window) > 0:
            yield (i - window_size + 1, tuple(window))
        else:
            return
        # Return windows
        for i, element in it:
            window.popleft()
            window.append(element)
            yield (i - window_size + 1, tuple(window))


class HTMLSequence(Sequence):
    """A sequence over HTML tags that always starts the window at an opening
    tag."""
    def __init__(self, html):
        self.tags = filter(
            lambda fragment: isinstance(fragment, hp.HtmlTag),
            html.parsed_body)
        super(HTMLSequence, self).__init__(
            [(x.tag, x.tag_type) for x in self.tags])

    @staticmethod
    def generic_close_tag(element):
        tag, tag_type = element
        return '/>' if tag_type == hp.HtmlTagType.CLOSE_TAG else tag

    def __iter__(self):
        return itertools.imap(HTMLSequence.generic_close_tag, self._iterable)

    def roll(self, window_size):
        def balanced(window):
            stack = []
            for tag, tag_type in window:
                if tag_type == hp.HtmlTagType.OPEN_TAG:
                    stack.append((tag, tag_type))
                elif tag_type == hp.HtmlTagType.CLOSE_TAG:
                    try:
                        last_tag, last_tag_type = stack.pop()
                    except IndexError:
                        return False
                    if (last_tag_type != hp.HtmlTagType.OPEN_TAG or
                        last_tag != tag):
                        return False
            return not stack

        return ((i, map(HTMLSequence.generic_close_tag, window))
                for (i, window) in super(HTMLSequence, self).roll(window_size)
                if balanced(window))


class TestSequence(Sequence):
    """Random sequence for testing."""
    def __init__(self, alphabet=['A', 'B', 'C'], motifs=['ABC'], m=0.2, size=100):
        """Generate a new random sequence.

        Parameters:
            - alphabet: a list of letters
            - motifs  : a list of strings
            - m       : probability of starting a motif at a random position
                        of the sequence
            - size    : length of the sequence
        """
        self.alphabet = set(alphabet)
        for motif in motifs:
            self.alphabet = self.alphabet.union(motif)
        self.alphabet = list(self.alphabet)
        self.motifs = motifs
        # Motif average length
        L = sum(len(motif) for motif in motifs)/float(len(motifs))
        p = m*L  # threshold to insert a new motif
        s = ''   # random string
        n = 0    # string length
        while n < size:
            motif = random.choice(motifs)
            w = len(motif)
            if random.random() <= p:
                s += motif
            else:
                for i in xrange(w):
                    s += random.choice(self.alphabet)
            n += w
        super(TestSequence, self).__init__(s[:size])

    def find_motif(self):
        """Return non-ovarlapping indices where the motifs appear.

        Note: motifs are searched in order. If a motif is found the remaining
        motifs are ignored.

        Example: consider motifs 'ABC' and 'BCAA' and the following sequence:

        AAABCAABBB
          ---   <- ABC
           ---- <- BCAA

        Since the motifs overlap, and ABC comes first then only the ABC match
        is returned.
        """
        s = self._iterable
        i = 0
        r = []
        while i < len(s):
            match = False
            for motif in self.motifs:
                w = len(motif)
                if s[i:i+w] == motif:
                    r.append(i)
                    i += w
                    match = True
                    break
            if not match:
                i += 1
        return r


class MEME(util.Logged):
    """Implementation of the MEME algorithm.

    MEME tries to find repeating patterns inside an stream of symbols. It was
    originally conceived to find patterns inside long molecule sequences
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
    """
    def __init__(self, logger='default'):
        super(MEME, self).__init__(logger)


    def _roll(self, sequence, W):
        return ((i, np.array([self.code_book.code(x) for x in window]))
                for i, window in sequence.roll(W))


    @staticmethod
    def _score(X, f):
        pM = 1.0 # probability of X, if taken from motif
        pB = 1.0 # probability of X, if taken from background
        for j, k in enumerate(X):
            pM *= f[j + 1, k]
            pB *= f[0    , k]
        return pM, pB


    def score(self, X, k=0):
        """Return the motif and background probabilities of seeing the
        sub-sequence X"""
        return MEME._score(map(self.code_book.code, X),
                           self.position_frequency_matrix[k])


    def _fit_1(self, sequence, V, f0, m0, eps, max_iter, relax):
        """Run the Expectation-Maximization algorithm.

        Parameters:
            - sequence: sequence to fit
            - V       : erasure parameters
            - f       : initial estimate of position frequency matrix
            - m       : initial estimate of a priori model probability
            - eps     : stop when relative change in expected log-likelihood
                        is less than this
            - relax   : parameter to take probabilities away from zero
        """
        W = f0.shape[0] - 1                    # size of window
        n = len(V)                             # sequence length
        it = 0                                 # current iteration
        f1 = f0                                # current estimation of f
        m1 = m0                                # current estimation of m
        E1 = None                              # current expected log-likelihood

        while True:
            pM = np.ones((n,))
            pB = np.ones((n,))
            for i, X in self._roll(sequence, W):
                pM[i], pB[i] = MEME._score(X, f1)
            pM *= m1
            pB *= 1.0 - m1

            Z = pM/(pM + pB)
            E2 = np.sum(Z*np.log(pM) + (1 - Z)*np.log(pB))
            # Correct for overlapping, enforce Z[i:(i+W)] <= 1 for i=1...n
            # MEME hack to account for dependence between sub-sequences.
            for i in xrange(len(Z)):
                s = Z[i:i+W].sum()
                if s > 1.0:
                    Z[i:i+W] /= s

            # Update m and f
            qB = 0.0
            qM = 0.0
            c = np.zeros(f0.shape, dtype=float)
            for i, X in self._roll(sequence, W):
                ZB = 1.0 - Z[i]
                ZM = Z[i]*V[i]
                for j, k in enumerate(X):
                    c[0    , k] += ZB
                    c[j + 1, k] += ZM
                qB += ZB
                qM += ZM
            m2 = qM/(qB + qM)
            f2 = (1.0 - relax)*util.normalized(c) + relax
            # Check convergence
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

            f1 = f2
            m1 = m2

        return f2, m2, E2, Z

    def _fit_2(self, sequence, m0, seeds, V, gamma, eps, max_iter, relax):
        """Call EM for the given seeds"""
        f2 = m2 = E2 = Z2 = None
        for s, seed in enumerate(seeds):
            f0 = util.guess_emissions(self.code_book, seed, gamma)
            f1, m1, E1, Z1 = self._fit_1(sequence, V, f0, m0, eps, max_iter, relax)
            if not E2 or E1 > E2:
                f2, m2, E2, Z2 = f1, m1, E1, Z1
        return f2, m2, E2, Z2


    def fit(self, sequence, window_min, window_max=None,
            n_motifs=1, gamma=0.1, alpha=0.01, eps=1e-5, max_iter=200, relax=1e-6):
        if window_max is None:
            window_max = window_min

        self.code_book = util.CodeBook(sequence)
        self.n_motifs = n_motifs

        A = len(self.code_book)         # alphabet size
        n = self.code_book.total_count  # sequence length

        W0 = 1
        logP0 = np.sum(
            self.code_book.frequencies*np.log(self.code_book.frequencies))

        self.em_objective = np.zeros((n_motifs,))
        self.position_frequency_matrix = []
        self.motif_probability = np.zeros((n_motifs,))

        # Erasure coefficients, which are updated after each pass
        U = np.ones((n,))
        V = np.ones((n,))
        for p in xrange(n_motifs):
            W3 = G3 = None
            f3 = m3 = E3 = Z3 = None
            for W in xrange(window_min, window_max + 1):
                I = [i for i, X in self._roll(sequence, W)]
                N = len(I)
                if N == 0:
                    self.logger.info('MEME.fit: no sequences of length {0}'.format(W))
                    continue
                f2 = m2 = E2 = Z2 = None
                for m0 in util.log_range(1.0/n, 1.0/W, 2.0):
                    # Number of seeds
                    Q = min(N,
                            max(1,
                                int(np.log(1.0 - alpha)/np.log(1.0 - m0))))
                    # Select subsequences as seeds
                    r = random.sample(I, Q)
                    seeds = [X for i, X in self._roll(sequence, W) if i in r]
                    f1, m1, E1, Z1 = self._fit_2(
                        sequence, m0, seeds, V, gamma, eps, max_iter, relax)
                    if self.logger:
                        self.logger.info(
                            'MEME.fit pass={0} width={1} m0={2:.4e} seeds={3} E={4:.4e} m={5:.4e}'.format(p, W, m0, Q, E1, m1))
                    if not E2 or E1 > E2:
                        f2, m2, E2, Z2 = f1, m1, E1, Z1

                G2 = util.model_score((W/W0)*N*logP0, E2, (W - W0)*(A - 1))
                if self.logger:
                    self.logger.info(
                        'MEME.fit E1={0} G={1}'.format(E2, G2))

                if not G3 or G2<G3:
                    W3 = W
                    G3 = G2
                    f3, m3, E3, Z3 = f2, m2, E2, Z2

                if G3 == 0:
                    self.logger.info('Switched null model')
                    G3 = 1.0
                    logP0 = E3/N
                    W0 = float(W3)

            # Save motif
            self.em_objective[p] = E3
            self.position_frequency_matrix.append(f3)
            self.motif_probability[p] = m3

            # Update erasure
            for i in xrange(n):
                U[i] *= 1.0 - V[i]*Z3[max(0, i - W3 + 1):(i + 1)].max()
            for i in xrange(n):
                V[i] = U[i:(i + W3)].min()


        self._motif_threshold = np.log((1.0 - self.motif_probability)/self.motif_probability)

    def motif_width(self, p):
        return self.position_frequency_matrix[p].shape[0] - 1

    def find_motif(self, sequence, p=0):
        n = sum(1 for _ in sequence)
        S = np.zeros((n,))
        W = self.motif_width(p)
        for i, X in sequence.roll(W):
            pM, pB = self.score(X, p)
            Z = np.log(pM/pB)
            S[i] = Z

        for i, X in sequence.roll(W):
            Z = S[i]
            if (Z == np.max(S[max(0, i - W):min(i + W, n)]) and
                Z > self._motif_threshold[p]):
                yield MotifMatch(
                    index=i,
                    group=p,
                    score=Z,
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
    motif = 'ABBA'
    def make_sequence(n, m=0.1):
        return TestSequence(
            size=n, motifs=[motif], m=m, alphabet=string.ascii_uppercase)

    seq_train = make_sequence(1000)
    meme = MEME()
    meme.fit(seq_train, window_min=2, window_max=8)

    W = meme.motif_width(0)
    print 'Motif length = {0} (true = {1})'.format(W, len(motif))
    print 'Parameters:'
    print '    m = {0}'.format(meme.motif_probability[0])
    print '    f = '
    print meme.position_frequency_matrix[0][1:,:]
    print 'True f:'
    tf = np.zeros((len(motif), len(seq_train.alphabet)))
    for i, letter in enumerate(motif):
        tf[i, meme.code_book.code(letter)] = 1.0
    print tf

    test_size = 10000
    seq_test = make_sequence(test_size)
    found = set(match.index for match in meme.find_motif(seq_test))
    real = set(seq_test.find_motif())

    cm = np.zeros((2,2))
    cm[0, 0] = len(found and real )
    cm[0, 1] = len(real  -   found )
    cm[1, 0] = len(found -   real)
    cm[1, 1] = test_size - W - cm.sum()
    print 'Confusion matrix:'

    print tabulate.tabulate(
        [[''      , 'Predicted M' , 'Predicted B' ],
         ['True M',  str(cm[0, 0]),  str(cm[0, 1])],
         ['True B',  str(cm[1, 0]),  str(cm[1, 1])]]
    )


def demo2():
    page = hp.url_to_page('https://news.ycombinator.com/')
    meme = MEME()
    s = HTMLSequence(page)
    meme.fit(s, 4, 50)
    for x in meme.find_motif(s, 0):
        print 80*'-'
        print page.body[
            s.tags[x.index].start:s.tags[x.index + meme.motif_width(0)].end]


if __name__ == '__main__':
    demo2()
