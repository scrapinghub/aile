import collections
import itertools

import numpy as np
import sklearn.cluster
import scrapely.htmlpage as hp

import aile.page_extractor as pe


def jaccard_index(s1, s2, null_val=1.0):
    """Compute Jaccard index between two sets"""
    if s1 or s2:
        I = float(len(s1 & s2))
        return I / (len(s1) + len(s2) - I)
    else:
        return null_val


def is_tag(fragment):
    """Check if a fragment is also an HTML tag"""
    return isinstance(fragment, hp.HtmlTag)


def get_class(fragment):
    """Return a set with class attributes for a given fragment"""
    if is_tag(fragment):
        return set((fragment.attributes.get('class') or '').split())
    else:
        return set()


def get_tag(fragment):
    if fragment.is_text_content:
        return '[T]'
    elif is_tag(fragment):
        return fragment.tag
    else:
        return None


class TreeNode(object):
    __slots__ = ('tag', 'class_attr')

    def __init__(self, tag, class_attr=set()):
        self.tag = tag
        self.class_attr = class_attr

    def __hash__(self):
        return hash(self.tag)

    def __eq__(self, other):
        return self.tag == other.tag

    def __str__(self):
        s = self.tag
        if self.class_attr:
            s += '['
            s += ','.join(self.class_attr)
            s += ']'
        return s

    def __repr__(self):
        return self.__str__()

    @classmethod
    def similarity(cls, a, b, no_class=1.0):
        return jaccard_index(
            set([a.tag]) | a.class_attr,
            set([b.tag]) | b.class_attr,
            no_class)


def non_empty_text(page, fragment):
    return fragment.is_text_content and\
        page.body[fragment.start:fragment.end].strip()


def fragment_to_node(page, fragment):
    """Convert a fragment to a node inside a tree where we are going
    to compute the kernel"""
    if non_empty_text(page, fragment):
        return TreeNode('[T]')
    elif (is_tag(fragment) and
          fragment.tag_type != hp.HtmlTagType.CLOSE_TAG):
        return TreeNode(fragment.tag, get_class(fragment))
    return None


def tree_nodes(page):
    """Return a list of fragments from page where empty text has been deleted"""
    for i, fragment in enumerate(page.parsed_body):
        node = fragment_to_node(page, fragment)
        if node is not None:
            yield (i, node)


class PageTree(object):
    def __init__(self, page):
        self.page = page
        index, self.nodes = zip(*tree_nodes(page))
        self.index = np.array(index)
        reverse_index = np.repeat(-1, len(page.parsed_body))
        for i, idx in enumerate(self.index):
            reverse_index[idx] = i
        match = pe.match_fragments(page.parsed_body)
        self.match = np.repeat(-1, len(self.index))
        self.parents = np.repeat(-1, len(self.index))
        for i, m in enumerate(match):
            j = reverse_index[i]
            if j >= 0:
                if m >= 0:
                    k = -1
                    while k < 0:
                        k = reverse_index[m]
                        m += 1
                        if m == len(match):
                            k = len(self.match)
                            break
                    assert k >= 0
                else:
                    k = j # no children
                self.match[j] = k
        for i, m in enumerate(self.match):
            self.parents[i+1:m] = i


    def __len__(self):
        return len(self.index)

    def children(self, i):
        return i + 1 + np.flatnonzero(self.parents[i+1:self.match[i]] == i)

    def similarity(self, i, j):
        return TreeNode.similarity(self.nodes[i], self.nodes[j])


def order_pairs(nodes):
    """Given a list of fragments return pairs of equal nodes ordered in
    such a way that if pair_1 comes before pair_2 then no element of pair_2 is
    a descendant of pair_1"""
    grouped = collections.defaultdict(list)
    for i, node in enumerate(nodes):
        grouped[node].append(i)
    return sorted(
        [pair
         for node, indices in grouped.iteritems()
         for pair in itertools.combinations_with_replacement(sorted(indices), 2)],
        key=lambda x: x[0], reverse=True)


def check_order(op, parents):
    """Test for order_pairs"""
    N = len(parents)
    C = np.zeros((N, N), dtype=int)
    for i, j in op:
        pi = parents[i]
        pj = parents[j]
        if pi > 0 and pj > 0:
            assert C[pi, pj] == 0
        C[i, j] = 1


def build_counts(ptree, max_depth=4, max_childs=20):
    N = len(ptree)
    if max_childs is None:
        max_childs = N
    pairs = order_pairs(ptree.nodes)
    C = np.zeros((N, N, max_depth), dtype=float)
    S = np.zeros((max_childs + 1, max_childs + 1, max_depth), dtype=float)
    S[0, :, :] = S[:, 0, :] = 1
    for i1, i2 in pairs:
        ch1 = ptree.children(i1)
        ch2 = ptree.children(i2)
        if len(ch1) == 0 and len(ch2) == 0:
            C[i2, i1, :] = C[i1, i2, :] = ptree.similarity(i1, i2)
        else:
            nc1 = min(len(ch1), max_childs)
            nc2 = min(len(ch2), max_childs)
            for j1 in range(1, nc1 + 1):
                for j2 in range(1, nc2 + 1):
                    S[j1, j2,  :]  = S[j1 - 1, j2    , :  ] +\
                                     S[j1    , j2 - 1, :  ] -\
                                     S[j1 - 1, j2 - 1, :  ]
                    S[j1, j2, 1:] += S[j1 - 1, j2 - 1, :-1]*C[ch1[j1 - 1], ch2[j2 - 1], :-1]
            C[i2, i1, :] = C[i1, i2, :] = \
                        ptree.similarity(i1, i2)*S[nc1, nc2, :]
    return C[:, :, max_depth - 1]


def kernel(ptree, counts=None, max_depth=2, max_childs=20, decay=0.1):
    if counts is None:
        C = build_counts(ptree, max_depth, max_childs)
    else:
        C = counts
    K = np.zeros(C.shape)
    N = K.shape[0]
    A = C.copy()
    B = C.copy()
    for i in range(N - 1, -1, -1):
        pi = ptree.parents[i]
        for j in range(N - 1, -1, -1):
            pj = ptree.parents[j]
            if pi > 0:
                A[pi, j] += decay*A[i, j]
            if pj > 0:
                B[i, pj] += decay*B[i, j]
    for i in range(N - 1, -1, -1):
        pi = ptree.parents[i]
        for j in range(N - 1, -1, -1):
            ri = max(ptree.match[i], i)
            rj = max(ptree.match[j], j)
            K[i, j] += A[i, j] + B[i, j] - C[i, j]
            pj = ptree.parents[j]
            if pi > 0 and pj > 0:
                K[pi, pj] += decay*K[i, j]
    return K


def to_rows(d):
    return np.tile(d, (len(d), 1))


def to_cols(d):
    return np.tile(d.reshape(len(d), -1), (1, len(d)))


def normalize_kernel(K):
    d = np.diag(K).copy()
    d[d == 0] = 1.0
    return K/np.sqrt(to_rows(d)*to_cols(d))


def kernel_to_distance(K):
    d = np.diag(K)
    return np.sqrt(to_rows(d) + to_cols(d) - 2*K)


def kernel_to_radial_distance(K):
    return -np.log(normalize_kernel(K))


def cluster(K):
    D = kernel_to_distance(normalize_kernel(K))
    clt = sklearn.cluster.DBSCAN(eps=0.76, min_samples=8, metric='precomputed')
    return clt.fit_predict(D)


def score_clusters(ptree, labels):
    grp = collections.defaultdict(list)
    for i, l in enumerate(labels):
        grp[l].append(i)
    grp = {k: np.array(v) for k, v in grp.iteritems()}
    scores = {k: sum(max(0, ptree.match[i] - i + 1) for i in v)
              for k, v in grp.iteritems()}
    return grp, scores
