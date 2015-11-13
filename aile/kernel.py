import collections

import numpy as np
import sklearn.cluster
import scrapely.htmlpage as hp

import aile.page_extractor as pe
import aile._kernel as _ker

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
        return frozenset((fragment.attributes.get('class') or '').split())
    else:
        return frozenset()


def get_tag(fragment):
    if fragment.is_text_content:
        return '[T]'
    elif is_tag(fragment):
        return fragment.tag
    else:
        return None


class TreeNode(object):
    __slots__ = ('tag', 'class_attr')

    def __init__(self, tag, class_attr=frozenset()):
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
            frozenset([a.tag]) | a.class_attr,
            frozenset([b.tag]) | b.class_attr,
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

    def children_matrix(self, max_childs=20):
        N = len(self.parents)
        C = np.repeat(-1, N*max_childs).reshape(N, max_childs)
        for i in range(N - 1, -1, -1):
            p = self.parents[i]
            if p >= 0:
                for j in range(max_childs):
                    if C[p, j] == -1:
                        C[p, j] = i
                        break
        return C

    def similarity(self, i1, i2):
        return TreeNode.similarity(self.nodes[i1], self.nodes[i2])


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


# Import cython functions
########################################################################
build_counts = _ker.build_counts
kernel = _ker.kernel
