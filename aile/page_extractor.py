import itertools
import bisect
import collections

import pandas as pd
import numpy as np
import scrapely.htmlpage as hp
import sklearn.cluster

import aile.util as util
from aile.phmm import ProfileHMM
from ptree import match_fragments


IGNORE_TAGS_DEFAULT = {'b', 'br', 'em',
                       'i', 'small', 'strong', 'sub', 'sup', 'wbr'}


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
        ignore_tags = IGNORE_TAGS_DEFAULT
    elif ignore_tags is None:
        ignore_tags = {}

    def get_class(fragment):
        class_attr = fragment.attributes.get('class')
        if class_attr is None:
            return []
        else:
            return class_attr.split()

    # Extract all classes
    class_count = collections.Counter(
        class_attr
        for fragment in page.parsed_body if isinstance(fragment, hp.HtmlTag)
        for class_attr in get_class(fragment))

    def convert(fragment):
        if (fragment.is_text_content and
            page.body[fragment.start:fragment.end].strip()):
            return PageSymbol(False, None, 'text')
        elif isinstance(fragment, hp.HtmlTag):
            if fragment.tag in ignore_tags:
                return PageSymbol(False, None, 'text')
            if fragment.tag_type != hp.HtmlTagType.CLOSE_TAG:
                class_attr = fragment.attributes.get('class')
                if class_attr:
                    class_attr = tuple(x for c, x in sorted(
                        (class_count[x], x) for x in class_attr.split())[-2:])
                return PageSymbol(False, class_attr, fragment.tag)
            else:
                return PageSymbol(True, None, fragment.tag)
        else:
            return None

    return list(
        join_text(
            filter(lambda x: x[0] is not None,
                   [(convert(f), f) for f in page.parsed_body])))


def build_tree(match):
    parents = np.repeat(-1, len(match))
    for i, j in enumerate(match):
        if j > i:
            parents[i+1:j] = i
    return parents


def extract_subtrees(fragments, w_min, w_max):
    """Extract all segments of web page that are balanced and have a size
    between w_min and w_max"""
    match = match_fragments(fragments)
    i = 0
    while i < len(match):
        j = match[i]
        if j > i:
            k = j
            while k - i <= w_max:
                if k - i >= w_min:
                    yield (i, k)
                if k < len(match) - 1:
                    k = match[k + 1]
                    if k < j:
                        break
                else:
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


def extract_motifs_2(model, X, prop=0.7, margin=0.5):
    Z, logP = model.viterbi(X)
    def motif(i, j):
        return Motif((i, j), Z[i:j],
                            model.score(X[i:j], Z[i:j])/(j - i), 0.0)
    i1 = None
    i2 = None
    z1 = None
    z2 = None
    for i, z in enumerate(Z):
        if z >= model.W:
            if z1 is None:
                i1 = i
                z1 = z
            elif z <= z2 - model.W*margin:
                if (z2 - z1) >= model.W*prop:
                    yield motif(i1, i2+1)
                i1 = i
                z1 = z
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


FitResult = collections.namedtuple(
    'FitResult',
    ['model', 'logP', 'code_book', 'extractors', 'motifs', 'items']
)


def fit_model(page_sequence):
    code_book = util.CodeBook(page_sequence.tags)
    X = np.array(map(code_book.code, page_sequence.tags))
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
            f = util.guess_emissions(code_book.frequencies, seed)
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
        subtrees = list(extract_subtrees(page_sequence.fragments, int(model.W*0.8), int(model.W*1.2)))
        motifs = list(extract_motifs_1(model, X, subtrees))
        model = adjust(model, motifs)
        model.fit_em_1(X, 1)
        motifs = list(extract_motifs_2(model, X))
        items, extractors = extract_items_2(page_sequence, motifs, model.W/2)
        empty = uninformative_fields(items)
        fields = [f for f in items.columns.levels[0] if f not in empty]
        extractors = [e for i, e in enumerate(extractors) if i not in empty]
        if not fields:
            print "Not fields for model", model.W
            continue
        items = pd.concat(
            [items[f] for f in fields],
            axis=1,
            keys=fields)
        res.append(FitResult(model, logP, code_book, extractors, motifs, items))
    return res


def fit_result_extract(model, code_book, extractors, page_sequence, min_prop=0.6):
    X = np.array(map(code_book.code, page_sequence.tags))
    motifs = list(extract_motifs_2(model, X))
    match = match_fragments(page_sequence.fragments)
    parents = build_tree(match)

    items = []
    for (i, j), Z, score_1, score_2 in motifs:
        row = []
        rgroup = rpath_group(relative_paths(parents, i, j), page_sequence.tags)
        success = 0
        for extractor in extractors:
            try:
                k = rgroup.index(extractor)
                cell = fragment_to_cell(
                    page_sequence, i + k, page_sequence.fragments[i + k])
                success += 1
            except ValueError:
                cell = (None, None, None)
            row += cell
        if float(success)/len(extractors) >= min_prop:
            items.append(row)
    fields = range(len(extractors))
    return pd.DataFrame.from_records(
        items,
        columns=index_fields(fields))


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
    bw = 2
    def distance(X, w):
        Z = np.ones(X.shape, dtype=int)
        for i in range(-bw, bw + 1):
            v = w + i
            Z[v:] = np.logical_and(Z[v:], X[:-v] != X[v:])
        return float(Z.sum())/(len(X) - w)

    D = [(distance(X, w), w)
                for w in np.arange(min_width, max_width)]
    ad, aw = zip(*D)
    estimates = []
    for d, w in sorted(D):
        if not estimates or abs(estimates[-1] - w) > 2*bw:
            estimates.append(w)
        if len(estimates) >= n_estimates:
            break
    return estimates


def index_fields(fields):
    N = len(fields)
    return [np.repeat(fields, 3),
            np.tile(['name', 'type', 'content'], N)]


def fragment_to_cell(page_sequence, i, fragment):
    if fragment is not None:
        if fragment.is_text_content:
            txt = page_sequence.body_segment(i)
            if txt.strip():
                return (None, 'txt', txt)
        elif (isinstance(fragment, hp.HtmlTag) and
              fragment.tag_type != hp.HtmlTagType.CLOSE_TAG):
            name = fragment.attributes.get('class', None)
            if fragment.tag == 'a':
                attr = u''
                href = fragment.attributes.get('href')
                title = fragment.attributes.get('title')
                if href:
                    attr += u'href="{0}" '.format(href)
                if title:
                    attr += u'title="{0}" '.format(title)
                return (name, 'lnk', attr)
            if fragment.tag == 'img':
                return (name, 'img',
                        u'src="{0}" alt="{1}"'.format(
                            fragment.attributes.get('src', ''),
                            fragment.attributes.get('alt', '')))
        return (None, None, None)


def extract_items_1(page_sequence, motifs, fields):
    items = []
    scores = []
    for (i, j), Z, score, H in motifs:
        f = {}
        for k, z in enumerate(Z):
            if z in fields:
                fragment = page_sequence.fragments[i + k]
                f[z] = fragment_to_cell(page_sequence, i + k, fragment)
        items.append([x for field in fields
                        for x in f.get(field, (None, None, None))])
        scores.append(score)
    return pd.DataFrame.from_records(
        items,
        columns=index_fields(fields)), np.array(scores)


def relative_paths(parents, i, j):
    paths = []
    for k in range(i, j):
        p = []
        q = k
        while q >= i:
            p.append(q)
            q = parents[q]
        paths.append(p)
    return paths


def rpath_walk(rpath, tags, max_depth=2, ignore_tags='default'):
    # Ignore stylistic tags
    if ignore_tags is 'default':
        ignore_tags = IGNORE_TAGS_DEFAULT
    elif ignore_tags is None:
        ignore_tags = {}
    N = len(rpath)
    if max_depth is None:
        max_depth = N
    i = 0
    walk = []
    while i < max_depth:
        symbol = tags[rpath[i]]
        if symbol.tag not in ignore_tags:
            walk.append(symbol)
            i += 1
        if i >= N:
            break
    return tuple(walk)


def rpath_group(rpaths, tags, max_depth=2, ignore_tags='default'):
    walks = [rpath_walk(rpath, tags, max_depth, ignore_tags)
             for rpath in rpaths]
    count = collections.defaultdict(int)
    group = []
    for walk in walks:
        group.append((walk, count[walk]))
        count[walk] += 1
    return group


def align_motifs(page_sequence, motifs):
    match = match_fragments(page_sequence.fragments)
    parents = build_tree(match)
    rpath_groups = [rpath_group(relative_paths(parents, i, j), page_sequence.tags)
                    for (i, j), Z, score_1, score_2 in motifs]
    code_book = util.CodeBook(rpath
                              for rpath_group in rpath_groups
                              for rpath in rpath_group)
    N = len(motifs)
    A = len(code_book)
    alignment = np.repeat(-1, N*A).reshape(N, A)
    for rgroup, motif in zip(rpath_groups, alignment):
        for i, rpath in enumerate(rgroup):
            motif[code_book.code(rpath)] = i
    return alignment, code_book


def filter_motifs(alignment, max_diff=10):
    clt = sklearn.cluster.DBSCAN(
        eps=float(max_diff)/alignment.shape[1], min_samples=4, metric='hamming')
    y = clt.fit_predict(alignment != -1)
    u, c = np.unique(y, return_counts=True)
    return y == u[np.argmax(c)]


def filter_columns(alignment, flag):
    return np.any(alignment[flag, :]!=-1, axis=0)


def extract_items_2(page_sequence, motifs, max_diff):
    alignment, code_book = align_motifs(page_sequence, motifs)
    rows = filter_motifs(alignment, max_diff)
    cols = np.nonzero(filter_columns(alignment, rows))[0]
    valid = alignment[rows, :]
    motifs = [motif for motif, is_valid in zip(motifs, rows) if is_valid]
    items = []
    for align, motif in zip(valid, motifs):
        row = []
        for c in cols:
            idx = align[c]
            if idx >= 0:
                i = motif.range[0] + idx
                cell = fragment_to_cell(
                    page_sequence, i, page_sequence.fragments[i])
            else:
                cell = (None, None, None)
            row += cell
        items.append(row)
    fields = np.arange(len(cols))
    return pd.DataFrame.from_records(
        items,
        columns=index_fields(fields)), map(code_book.decode, cols)


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
