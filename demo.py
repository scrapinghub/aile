import numpy as np
import scrapely.htmlpage as hp

from phmm import ProfileHMM
import util

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
    priors = []
    # candidates start with a non-closing tag and end with a closing one
    candidates = [i for i in range(len(X) - W) if X[i] != s and X[i + W] == s]
    for i in util.random_pick(candidates, n):
        f = util.guess_emissions(code_book, X[i:i+W])
        f[1, s] = 0.0 # zero probability of starting motif with closing tag
        f[W, :] = 0.0 # zero probability of ending motif with non-closing tag
        f[W, s] = 1.0 # probability 1 of ending motif with closing tag
        emissions.append(util.normalized(f))
        eps = f[0, :].repeat(W).reshape(f[1:,:].shape)
        eps[  0, s] = 1e-6
        eps[W-1, :] = 1e-6
        eps[W-1, s] = 1.0
        priors.append(1.0 + 1e-3*util.normalized(eps))
    return emissions, priors


def demo2():
    page = hp.url_to_page('http://news.ycombinator.com')
    tags_1 = [fragment for fragment in page.parsed_body if isinstance(fragment, hp.HtmlTag)]
    tags_2 = [
        fragment.tag if fragment.tag_type != hp.HtmlTagType.CLOSE_TAG
        else '/>' for fragment in tags_1
    ]

    X = np.array(tags_2)
    phmm = ProfileHMM.fit(X, 10, 50, guess_emissions=html_guess_emissions)
    for (i, j), Z in phmm.extract(X):
        print 80*'#'
        print Z
        print 80*'-'
        print 'SCORE: ', phmm.score(
            np.array([phmm.code_book.code(x) for x in X[i:j]]), Z)
        print page.body[tags_1[i].start:tags_1[j].end]
    return phmm


if __name__ == '__main__':
    phmm = demo2()
