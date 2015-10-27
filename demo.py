import os
import sys
import heapq
import codecs
import itertools
import bisect
import collections
import urllib

import pandas as pd
import numpy as np
import scipy.spatial.distance as dst
import scrapely.htmlpage as hp

from aile.phmm import ProfileHMM
import aile.util as util


def phmm_cmp(W, Z1, Z2):
    return ((Z1 >= W) != (Z2 >= W)).mean()


def demo1():
    phmm_true = ProfileHMM(
        f=np.array([
            [0.2, 0.3, 0.2, 0.3],
            [0.9, 0.1, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2]]),
        t=np.array([
            0.05, 0.9, 0.1, 0.05, 0.85, 0.1])
    )

    X, Z = phmm_true.generate(5000)
    phmm = ProfileHMM.fit(X, 3)

    print "True model 't' parameters", phmm_true.t
    print " Estimated 't' paramaters", phmm.t

    z, logP = phmm.viterbi(X)
    print 'Error finding motifs (% mismatch):', phmm_cmp(phmm.W, Z, z)*100


PageSymbol = collections.namedtuple(
    'PageSymbol',
    ['closed', 'class_attr', 'tag']
)


def join_text(symbols_fragments):
    """If two or more text symbols are adjacent join them in a single one"""
    s = None
    f = None
    for symbol, fragment in symbols_fragments:
        if symbol.tag != 'text':
            if f is not None:
                yield s, f
                s = None
                f = None
            yield symbol, fragment
        else:
            if f is None:
                f = fragment
                s = symbol
            else:
                f.end = fragment.end
    if f is not None:
        yield s, f


def encode_html(page, ignore_tags='default'):
    """Given an HtmlPage transform it into a list of PageSymbol.

    Returns a list of tuples: the first element the PageSymbol and the
    second element the fragment that originated the symbol.
    """
    # Ignore stylistic tags
    if ignore_tags is 'default':
        ignore_tags = {'b', 'br', 'em',
                       'i', 'small', 'strong', 'sub', 'sup', 'wbr'}
    elif ignore_tags is None:
        ignore_tags = {}

    def convert(fragment):
        if (fragment.is_text_content and
            page.body[fragment.start:fragment.end].strip()):
            return PageSymbol(False, None, 'text')
        elif isinstance(fragment, hp.HtmlTag):
            if fragment.tag in ignore_tags:
                return None
            if fragment.tag_type != hp.HtmlTagType.CLOSE_TAG:
                return PageSymbol(False,
                                  fragment.attributes.get('class'),
                                  fragment.tag)
            else:
                return PageSymbol(True, None, None)
        else:
            return None

    return list(
        join_text(
            filter(lambda x: x[0] is not None,
                   [(convert(f), f) for f in page.parsed_body])))


def match_fragments(fragments):
    """Find the closing fragment for every fragment.

    Returns: an array with as many elements as fragments. If the
    fragment has no closing pair then the array contains -1 at that position
    otherwise it contains the index of the closing pair.
    """
    match = np.repeat(-1, len(fragments))
    stack = []
    for i, fragment in enumerate(fragments):
        if isinstance(fragment, hp.HtmlTag):
            if fragment.tag_type == hp.HtmlTagType.OPEN_TAG:
                stack.append((i, fragment))
            elif (fragment.tag_type == hp.HtmlTagType.CLOSE_TAG and
                  stack):
                last_i, last_tag = stack[-1]
                if (last_tag.tag_type == hp.HtmlTagType.OPEN_TAG and
                    last_tag.tag == fragment.tag):
                    match[last_i] = i
                    stack.pop()
    return match


def extract_subtrees(fragments, w_min, w_max):
    """Extract all segments of web page that are balanced and have a size
    between w_min and w_max"""
    match = match_fragments(fragments)
    i = 0
    while i < len(match):
        j = match[i]
        if j > 0:
            k = j
            while k - i <= w_max:
                if k - i >= w_min:
                    yield (i, k)
                k = match[k + 1]
                if k < j:
                    break
        i += 1


Motif = collections.namedtuple('Motif',
                               ['range', 'states', 'score_1', 'score_2'])


def extract_motifs_1(model, X, subtrees, G=0.5):
    """Filter the subtrees that are motifs according to model"""
    Z, logP = model.viterbi(X)
    k = 0
    for i, j in subtrees:
        if i > k:
            H = np.sum(
                    np.abs(np.bincount(Z[i:j], minlength=model.S)[model.W:] - 1)
                )/float(model.W)
            if H <= G:
                score = model.score(X[i:j], Z[i:j])/(j - i)
                yield Motif((i, j), Z[i:j], score, H)
                k = j


def extract_motifs_2(model, X, prop=0.7):
    Z, logP = model.viterbi(X)
    def motif(i, j):
        return Motif((i, j), Z[i:j],
                            model.score(X[i:j], Z[i:j])/(j - i), 0.0)
    i1 = None
    i2 = None
    z2 = None
    for i, z in enumerate(Z):
        if z >= model.W:
            if z2 is None:
                z2 = z
                i1 = i
            elif z <= z2:
                if (i2 - i1) >= model.W*prop:
                    yield motif(i1, i2+1)
                i1 = i
            i2 = i
            z2 = z
    if i1 is not None and (i2 - i1) >= model.W*prop:
        yield motif(i1, i2+1)


class PageSequence(object):
    def __init__(self, pages):
        """Concatenation of several pages"""
        self.pages = pages
        self.body_lengths = []
        self.tags_lengths = []
        self.tags = []
        self.fragments = []
        l1 = 0
        l2 = 0
        for page in pages:
            tags, fragments = zip(*encode_html(page))
            self.tags += tags
            self.fragments += fragments
            l1 += len(page.body)
            l2 += len(tags)
            self.body_lengths.append(l1)
            self.tags_lengths.append(l2)
        self.body = ''.join(page.body for page in pages)
        self.code_book = util.CodeBook(self.tags)

    def index_tag(self, i):
        return bisect.bisect(self.tags_lengths, i)

    def offset(self, fragment_idx):
        p = self.index_tag(fragment_idx)
        if p > 0:
            off = self.body_lengths[p - 1]
        else:
            off = 0
        return off

    def body_segment(self, fragment_idx_1, fragment_idx_2=None):
        off1 = self.offset(fragment_idx_1)
        fragment1 = self.fragments[fragment_idx_1]
        if fragment_idx_2 is None:
            return self.body[off1 + fragment1.start:off1 + fragment1.end]
        else:
            off2 = self.offset(fragment_idx_2)
            fragment2 = self.fragments[fragment_idx_2]
            return self.body[off1 + fragment1.start:off2 + fragment2.end]


def fit_model(page_sequence):
    X = np.array(map(page_sequence.code_book.code, page_sequence.tags))
    W = guess_motif_width(X, n_estimates=2)

    def html_guess_emissions(W, n_seeds):
        seeds = []
        w = W
        while (len(seeds) < n_seeds):
            seeds += [X[i:i+W]
                      for i, j in extract_subtrees(page_sequence.fragments, w, w)]
            w += 1
            if (w - W > 10):
                break
        if seeds:
            seeds = itertools.islice(itertools.cycle(seeds), n_seeds)
        else:
            seeds = [X[i:i+W] for i in
                     util.random_pick(range(len(X) - W - 1), n_seeds)]
        emissions = []
        priors = []
        for seed in seeds:
            f = util.guess_emissions(page_sequence.code_book.frequencies, seed)
            eps = f[0, :]
            emissions.append(f)
            priors.append(1.0 + 1e-3*eps)
        return emissions, priors

    models = ProfileHMM.fit(
        X,
        W,
        guess_emissions=html_guess_emissions,
        precision=1e-3)

    res = []
    for model, logP in models:
        f_inc, r_inc = increase_states(model.f)
        model = ProfileHMM(
            f=f_inc, t=model.t, p0=np.repeat(model.p0, r_inc), eps=model.eps)
        model.fit_em_n(X, 3)

        subtrees = list(extract_subtrees(page_sequence.fragments, int(model.W*0.8), int(model.W*1.2)))
        motifs = list(extract_motifs_1(model, X, subtrees))
        model = adjust(model, motifs)
        motifs = list(extract_motifs_2(model, X))

        fields = itemize(model, page_sequence.code_book)
        items, scores = extract_items(page_sequence, motifs, fields)
        valid = scores >= (np.median(scores) - 1.0)
        items = items.ix[valid]
        empty = uninformative_fields(items)
        fields = [f for f in fields if f not in empty]
        if not fields:
            continue
        items = pd.concat(
            [items[f] for f in fields],
            axis=1,
            keys=fields)
        res.append((model, logP, fields, motifs, items))
    return res


def adjust(phmm, motifs):
    r = np.zeros((phmm.W, ), dtype=int)
    for (i, j), Z, score, H in motifs:
        for k, z in enumerate(Z):
            if z >= phmm.W:
                r[z - phmm.W] += 1
                break
    start = np.argmax(r)
    phmm2 = ProfileHMM(
        f   = np.vstack((
                phmm.f[ 0, :],
                np.roll(phmm.f[1:phmm.W + 1 , :], -start, axis=0),
                np.roll(phmm.f[  phmm.W + 1:, :], -start, axis=0))),
        t   = phmm.t,
        eps = phmm.eps,
        p0  = phmm.p0)
    return phmm2


def increase_states(f, max_p=0.1):
    W = f.shape[0] - 1
    g = [f[0,:]]
    r = np.zeros((W,), dtype=int)
    for j in range(W):
        h = f[1 + j,:].copy()
        d = np.flatnonzero(h >= max_p)
        if len(d) > 1:
            for p in range(len(d)):
                for q in range(len(d)):
                    if q != p:
                        h[q] = 1e-12
                g.append(h)
                r[j] += 1
        else:
            g.append(h)
            r[j] += 1
    return np.vstack(g), np.tile(r, 2)


def itemize(phmm, code_book, ratio=2.0, tags=['text', 'a', 'img']):
    a = [i
         for i, symbol in enumerate(code_book.symbols)
         if symbol.tag in tags]
    h = phmm.f[0,a]
    return [phmm.W + j for j, g in enumerate(phmm.f[1:])
            if np.any(g[a]/h >= ratio)]


def guess_motif_width(X, n_estimates=2, min_width=10, max_width=400):
    D = [(dst.hamming(X[:-w], X[w:]), w)
         for w in np.arange(min_width, max_width)]
    return [w for d, w in heapq.nsmallest(n_estimates, D, key=lambda x: x[0])]


def index_fields(fields):
    N = len(fields)
    return [np.repeat(fields, 3),
            np.tile(['name', 'type', 'content'], N)]


def extract_items(page_sequence, motifs, fields):
    items = []
    scores = []
    for (i, j), Z, score, H in motifs:
        f = {}
        for k, z in enumerate(Z):
            if z in fields:
                fragment = page_sequence.fragments[i + k]
                if fragment is not None:
                    if fragment.is_text_content:
                        txt = page_sequence.body_segment(i + k)
                        if txt.strip():
                            f[z] = (None, 'txt', txt)
                    elif (isinstance(fragment, hp.HtmlTag) and
                          fragment.tag_type != hp.HtmlTagType.CLOSE_TAG):
                        name = fragment.attributes.get('class', None)
                        if fragment.tag == 'a':
                            f[z] = (name, 'lnk', fragment.attributes.get('href', None))
                        if fragment.tag == 'img':
                            f[z] = (name, 'img', fragment.attributes.get('src', None))
        items.append([x for field in fields
                        for x in f.get(field, (None, None, None))])
        scores.append(score)
    return pd.DataFrame.from_records(
        items,
        columns=index_fields(fields)), np.array(scores)


def biggest_group(series):
    s = series[series.notnull()]
    if len(s) == 0:
        return (0, None)
    return max(((v, k)
                for k, v in collections.Counter(s).iteritems()),
               key=lambda x: x[0])


def items_score(items):
    S = 0
    C = 0
    n = 0
    for col in items.columns.levels[0]:
        S += entropy_categorical(items[col]['name'])
        S += entropy_categorical(items[col]['type'])
        C += items[col]['content'].notnull().sum()
        n += 2
    return S/float(n), 3.0*float(C)/(items.shape[0]*items.shape[1])


def uninformative_fields(items):
    def null_proportion(s):
        return (float(s.isnull().sum())/
                float(len(s)))
    return {col
            for col in items.columns.levels[0]
            if null_proportion(items[col]['content'])>=0.95}


def entropy_categorical(X):
    _, c = np.unique(X, return_counts=True)
    t = np.sum(c)
    p = c.astype(float)/float(t)
    return -np.sum(p*np.log(p))


def train_test(pattern, start, end):
    return ([pattern.format(i) for i in range(start, end + 1)],
            pattern.format(end + 1))


def train_test_1(n_train=2):
    return ('hn',
            train_test('https://news.ycombinator.com/news?p={0}', 1, n_train + 1))


def train_test_2(n_train=1):
    return ('patchofland',
            train_test('https://patchofland.com/investments/page/{0}.html', 1, n_train + 1))


def train_test_3(n_train=6):
    return ('ebay',
            train_test('http://www.ebay.com/sch/Tires-/66471/i.html?_pgn={0}', 1, n_train + 1))


def train_test_4(n_train=6):
    return ('monster',
            train_test('http://jobsearch.monster.co.uk/browse/?pg={0}&re=nv_gh_gnl1147_%2F', 1, n_train + 1))


def train_test_5(n_train=3):
    pattern = 'http://lambda-the-ultimate.org/node?from={0}'
    return ('lambda', ([pattern.format(i) for i in range(0, n_train*10, 10)],
                       pattern.format(n_train*10)))


def train_test_6(n_train=3):
    return ('arstechnica',
            train_test('http://arstechnica.com/page/{0}/', 1, n_train + 1))


def download(train_test):
    root, (train, test) = train_test
    train_download = ['{0}-{1}.html'.format(root, i) for i in range(len(train))]
    for url, local in zip(train, train_download):
        if not os.path.exists(local):
            urllib.urlretrieve(url, local)
    test_download = '{0}-{1}.html'.format(root, len(train) + 1)
    if not os.path.exists(test_download):
        urllib.urlretrieve(test, test_download)
    return (map(make_local_url, train_download),
            make_local_url(test_download))


def make_local_url(path):
    return 'file:///' + os.path.abspath(path)


def demo2(train_test, out='demo'):
    train_urls, test_url = download(train_test)
    train = PageSequence([hp.url_to_page(url) for url in train_urls])
    models = fit_model(train)
    for model, logP, fields, motifs, items in models:
        outf = codecs.open(
            '{0}-{1}.html'.format(out, model.W), 'w', encoding='utf-8')
        items.to_html(outf)
        print model.W, logP, items_score(items), model.motif_entropy
    return train, models

if __name__ == '__main__':
    tests = [
        train_test_1,
        train_test_2,
        train_test_3,
        train_test_4,
        train_test_5,
        train_test_6
    ]

    n_test = int(sys.argv[1])
    train, phmm = demo2(tests[n_test-1](), out='demo-{0}'.format(n_test))
