import collections
import logging
import random

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats


def eq_delta(a, b, eps=1e-3):
    """True if |a - b| <= eps"""
    return np.all(np.logical_and(b - eps <= a, a <= b + eps))


def safe_log(x, eps=1e-100):
    """Logarithm that returns a value even when evaluated at 0"""
    return np.log(np.where(x < eps, eps, x))


def normalized(P):
    """Return an stochastic matrix (all rows sum to 1), proportional to P"""
    return (P.T/P.sum(axis=1)).T


def log_range(start, stop, step, endpoint=True):
    """From start to stop, multiplying with step.

    Example:
        print list(log_range(1.5, 5.0, 2.0))

        Output: [1.5, 3.0, 6.0]
    """
    x = start
    while x < stop:
        yield x
        x *= step
    if endpoint:
        yield x


def random_pick(x, n=1):
    """Pick a random element from x"""
    return [x[random.randint(0, len(x) - 1)] for i in xrange(n)]


def guess_emissions(code_book, X, gamma=0.1):
    """Given a sequence X that has been encoded with code_book guess the
    initial value of the emission matrix used in MEME and ProfileHMM"""
    A = len(code_book)
    W = len(X)
    def g(m):
        p = np.log(1.0/float(A))
        return (1.0 - m)*np.log((1.0 - m)/(A - 1)) - p + m*np.log(m) + gamma*p
    m = opt.newton(g, 0.95)
    # Position Frequency Matrix
    # columns    : one for each code_book letter
    # row 0      : background frequency
    # row 1 .. W : motif frequency
    f = np.zeros((W + 1, A))
    f[0, :] = code_book.frequencies
    for j in xrange(W):
        f[j + 1, :   ] = (1.0 - m)/(A - 1)
        f[j + 1, X[j]] = m
    return f


def model_score(logP0, logP1, fp):
    """Compare model 1 against model 0, where fp is the difference in free
    parameters. Returns a value between 0 (model 1 is better) and
    1 (model 0 is better)"""
    return stats.chi2.sf(2.0*(logP1 - logP0), fp)**(1.0/fp)


class Logged(object):
    """An object with a logger attribute"""
    def __init__(self, logger='default'):
        if logger is 'default':
            logging.basicConfig()
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger


class Categorical(object):
    def __init__(self, p):
        """Categorical distribution with probabilities 'p'"""
        self.p = p/p.sum()
        self.w = self.p.cumsum()

    def sample(self, n=1):
        r = np.random.rand(n)
        s = np.zeros((n,), dtype=int)
        a = np.repeat(False, n)
        for i, x in enumerate(self.w):
            b = r <= x
            s[np.logical_and(
                np.logical_not(a),
                b)] = i
            a = b
        return s


class CodeBook(object):
    """
    Map symbols to integers, allowing to code and decode sequences using
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
