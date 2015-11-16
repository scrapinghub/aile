import collections
import itertools

import numpy as np
import sklearn.cluster
import scrapely.htmlpage as hp
import networkx as nx

import aile.page_extractor as pe
import aile._kernel as _ker


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
        return i + 1 + np.flatnonzero(self.parents[i+1:max(i+1, self.match[i])] == i)

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

    def siblings(self, i):
        p = self.parents[i]
        if p != -1:
            return self.children(p)
        else:
            return np.flatnonzero(self.parents == -1)

    def all_paths(self, i):
        paths = []
        for j in range(i, max(i+1, self.match[i])):
            p = j
            path = []
            while p >= i:
                path.append(p)
                p = self.parents[p]
            paths.append(path)
        return paths

    def tree_size(self):
        r = np.arange(len(self.match))
        s = r + 1
        return np.where(s > self.match, s, self.match) - r

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


def tree_size_distance(page_tree):
    s = page_tree.tree_size()
    a = to_cols(s).astype(float)
    b = to_rows(s).astype(float)
    return np.abs(a - b)/(a + b)


def cluster(page_tree, K, d1=1.0, d2=1.0, eps=0.76, min_samples=8):
    clt = sklearn.cluster.DBSCAN(
        eps=eps, min_samples=min_samples, metric='precomputed')
    return clt.fit_predict(
        d1*kernel_to_distance(normalize_kernel(K)) +
        d2*tree_size_distance(page_tree))


def extract_trees(ptree, labels):
    labels = labels.copy()
    for i, l in enumerate(labels):
        if l != -1:
            labels[i:max(i, ptree.match[i])] = l
    scores = collections.defaultdict(int)
    for i, l in enumerate(labels):
        if l != -1:
            scores[l] += max(0, ptree.match[i] - i + 1)
    max_s, max_l = max((s, l) for (l, s) in scores.iteritems())
    trees = []
    i = 0
    while i < len(labels):
        children = ptree.children(i)
        if np.any(labels[children] == max_l):
            first = None
            item = []
            for c in children:
                m = labels[c]
                if m != -1:
                    if first is None:
                        first = m
                    elif m == first:
                        trees.append(item)
                        item = []
                    item.append(c)
            if item:
                trees.append(item)
            i = ptree.match[i]
        else:
            i += 1
    return trees


def dtw(D):
    m = D.shape[0]
    n = D.shape[1]
    DTW = np.zeros((m + 1, n + 1))
    DTW[:, 0] = np.inf
    DTW[0, :] = np.inf
    DTW[0, 0] = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            DTW[i, j] = D[i - 1, j - 1] + min(
                DTW[i - 1, j    ],
                DTW[i    , j - 1],
                DTW[i - 1, j - 1])
    return DTW


def dtw_path(DTW):
    m = DTW.shape[0] - 1
    n = DTW.shape[1] - 1
    i = m - 1
    j = n - 1
    s = np.zeros((m,), dtype=int)
    t = np.zeros((n,), dtype=int)
    while i >= 0 or j >= 0:
        s[i] = j
        t[j] = i
        if DTW[i, j + 1] < DTW[i + 1, j]:
            if DTW[i, j + 1] < DTW[i, j]:
                i -= 1
            else:
                i -= 1
                j -= 1
        elif DTW[i + 1, j] < DTW[i, j]:
            j -= 1
        else:
            i -= 1
            j -= 1
    return s, t


def dtw_match_1(s, t, D):
    s = s.copy()
    for i, j in enumerate(s):
        m = k = i
        d = D[i, j]
        while k < len(s) and s[k] == j:
            if D[k, j] < d:
                m = k
                d = D[k, j]
            k += 1
        k = i
        while k < len(s) and s[k] == j:
            if k != m:
                s[k] = -1
            k += 1
    return s


def dtw_match_2(s, t, D):
    return dtw_match_1(t, s, D.T)


def extract_items(ptree, trees, labels):
    all_paths = []
    all_nodes = []
    for tree in trees:
        paths = []
        nodes = []
        for root in tree:
            for path in ptree.all_paths(root):
                paths.append(labels[path].tolist())
                nodes.append(path[0])
        all_paths.append(paths)
        all_nodes.append(nodes)
    G = nx.Graph()
    for (p1, n1), (p2, n2) in itertools.combinations(
            zip(all_paths, all_nodes), 2):
        N1 = len(p1)
        N2 = len(p2)
        D = np.zeros((N1, N2))
        for i in range(N1):
            q1 = p1[i]
            for j in range(N2):
                q2 = p2[j]
                D[i, j] = max(len(q1), len(q2))
                for a, b in zip(q1, q2):
                    if a != b:
                        break
                    D[i, j] -= 1
        DTW = dtw(D)
        a1, a2 = dtw_path(DTW)
        m = dtw_match_1(a1, a2, D)
        for i, j in enumerate(m):
            if j != -1:
                G.add_edge(n1[i], n2[j])

    cliques = []
    for K in nx.find_cliques(G):
        if len(K) >= 0.5*len(trees):
            cliques.append(K)
    cliques.sort(reverse=True, key=lambda x: len(x))
    node_to_clique = {}
    for i, K in enumerate(cliques):
        for node in K:
            if node not in node_to_clique:
                node_to_clique[node] = i
    n_cols = max(node_to_clique.values()) + 1
    items = np.zeros((len(trees), n_cols)) - 1
    for i, tree in enumerate(trees):
        children = []
        for root in tree:
            children += range(root, max(root + 1, ptree.match[root]))
        for c in children:
            col = node_to_clique.get(c)
            if col:
                items[i, col] = c
    return items


# Import cython functions
########################################################################
build_counts = _ker.build_counts
kernel = _ker.kernel
