import sys
import heapq
import codecs
import itertools
import bisect
import collections

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


def join_text(tags_fragments):
    f = None
    for tag, fragment in tags_fragments:
        if tag != '[T]':
            if f is not None:
                yield '[T]', f
                f = None
            yield tag, fragment
        else:
            if f is None:
                f = fragment
            else:
                f.end = fragment.end
    if f is not None:
        yield '[T]', f


def encode_html(page):
    ignore_tags = {'b', 'br', 'em',
                   'i', 'small', 'strong', 'sub', 'sup', 'wbr'}
    def convert(fragment):
        if (fragment.is_text_content and
            page.body[fragment.start:fragment.end].strip()):
            return '[T]'
        elif isinstance(fragment, hp.HtmlTag):
            if fragment.tag in ignore_tags:
                return None
            if fragment.tag_type != hp.HtmlTagType.CLOSE_TAG:
                tag_class = fragment.attributes.get('class', None)
                if tag_class:
                    return fragment.tag + '|' + tag_class
                else:
                    return fragment.tag
            else:
                return '/>'
        else:
            return None
    return list(
        join_text(
            filter(lambda x: x[0] is not None,
                   [(convert(f), f) for f in page.parsed_body])))


def match_fragments(fragments):
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


def extract_motifs(phmm, X, subtrees, G=0.3):
    Z, logP = phmm.viterbi(X)
    k = 0
    for i, j in subtrees:
        if i > k:
            H = np.sum(
                    np.abs(np.bincount(Z[i:j], minlength=phmm.S)[phmm.W:] - 1)
                )/float(phmm.W)
            if H <= G:
                score = phmm.score(X[i:j], Z[i:j])/(j - i)
                yield (i, j), Z[i:j], score, H
                k = j


class PageSequence(object):
    def __init__(self, pages):
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
        s = page_sequence.code_book.code('/>') # integer for the closing tag symbol
        emissions = []
        priors = []
        for seed in seeds:
            f = util.guess_emissions(page_sequence.code_book.frequencies, seed)
            f[1, s] = 0.0 # zero probability of starting motif with closing tag
            f[W, :] = 0.0 # zero probability of ending motif with non-closing tag
            f[W, s] = 1.0 # probability 1 of ending motif with closing tag
            f = util.normalized(f)
            eps = f[0, :].repeat(W).reshape(f[1:,:].shape)
            eps[  0, s] = 1e-6
            eps[W-1, :] = 1e-6
            eps[W-1, s] = 1.0
            emissions.append(f)
            priors.append(1.0 + 1e-3*util.normalized(eps))
        return emissions, priors

    models = ProfileHMM.fit(
        X,
        W,
        guess_emissions=html_guess_emissions,
        precision=1e-3)

    res = []
    for model, logP in models:
        subtrees = list(extract_subtrees(page_sequence.fragments, int(model.W*0.8), int(model.W*1.2)))
        motifs = list(extract_motifs(model, X, subtrees))
        model = adjust(model, motifs)
        model.fit_em_n(X, 3)
        motifs = list(extract_motifs(model, X, subtrees))
        fields = itemize(model, page_sequence.code_book)
        items = extract_items(page_sequence, motifs, fields)
        empty = uninformative_fields(items)
        fields = [f for f in fields if f not in empty]
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


def itemize(phmm, code_book, ratio=2.0, tags=['[T]', 'a', 'img']):
    a = []
    for i, letter in enumerate(code_book.letters):
        for tag in tags:
            if letter == tag or letter.startswith(tag + '|'):
                a.append(i)
                break
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
    return pd.DataFrame.from_records(
        items,
        columns=index_fields(fields))


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
    return S/float(n), C


def uninformative_fields(items):
    return {col
            for col in items.columns.levels[0]
            if items[col]['content'].isnull().all()}


def entropy_categorical(X):
    _, c = np.unique(X, return_counts=True)
    t = np.sum(c)
    p = c.astype(float)/float(t)
    return -np.sum(p*np.log(p))


def train_test(pattern, start, end):
    return ([pattern.format(i) for i in range(start, end + 1)],
            pattern.format(end + 1))


def train_test_1(n_train=2):
    return train_test('https://news.ycombinator.com/news?p={0}', 1, n_train + 1)


def train_test_2(n_train=1):
    return train_test('https://patchofland.com/investments/page/{0}.html', 1, n_train + 1)


def train_test_3(n_train=6):
    return train_test('http://www.ebay.com/sch/Tires-/66471/i.html?_pgn={0}', 1, n_train + 1)


def train_test_4(n_train=6):
    return train_test('http://jobsearch.monster.co.uk/browse/?pg={0}&re=nv_gh_gnl1147_%2F', 1, n_train + 1)


def train_test_5(n_train=3):
    pattern = 'http://lambda-the-ultimate.org/node?from={0}'
    return ([pattern.format(i) for i in range(0, n_train*10, 10)],
            pattern.format(n_train*10))


def train_test_6(n_train=3):
    return train_test('http://arstechnica.com/page/{0}/', 1, n_train + 1)


def demo2(train_test, out='demo'):
    train_urls, test_url = train_test
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
    phmm = demo2(tests[n_test-1](), out='demo-{0}'.format(n_test))
