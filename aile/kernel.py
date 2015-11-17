import collections
import itertools

import numpy as np
import sklearn.cluster
import networkx as nx

import _kernel as _ker
import dtw


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


def path_distance(p1, p2):
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
    return D


def find_cliques(G, min_size):
    cliques = []
    for K in nx.find_cliques(G):
        if len(K) >= min_size:
            cliques.append(set(K))
    cliques.sort(reverse=True, key=lambda x: len(x))
    L = set()
    for K in cliques:
        K -= L
        L |= K
    cliques = [J for J in cliques if len(J) >= min_size]
    node_to_clique = {}
    for i, K in enumerate(cliques):
        for node in K:
            if node not in node_to_clique:
                node_to_clique[node] = i
    return node_to_clique


def paths_and_nodes(ptree, trees, labels):
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
    return all_paths, all_nodes


def match_graph(all_paths, all_nodes):
    G = nx.Graph()
    for (p1, n1), (p2, n2) in itertools.combinations(
            zip(all_paths, all_nodes), 2):
        D = path_distance(p1, p2)
        DTW = dtw.from_distance(D)
        a1, a2 = dtw.path(DTW)
        m = dtw.match(a1, a2, D)
        for i, j in enumerate(m):
            if j != -1:
                G.add_edge(n1[i], n2[j])
    return G


def align_items(ptree, trees, node_to_clique):
    n_cols = max(node_to_clique.values()) + 1
    items = np.zeros((len(trees), n_cols), dtype=int) - 1
    for i, tree in enumerate(trees):
        for root in tree:
            for c in range(root, max(root + 1, ptree.match[root])):
                try:
                    items[i, node_to_clique[c]] = c
                except KeyError:
                    pass
    return items


def extract_items(ptree, trees, labels):
    return align_items(
        ptree,
        trees,
        find_cliques(
            match_graph(*paths_and_nodes(ptree, trees, labels)),
            0.5*len(trees))
    )


class ItemExtract(object):
    def __init__(self, page_tree):
        self.page_tree = page_tree
        self.kernel = _ker.kernel(page_tree)
        self.labels = cluster(page_tree, self.kernel)
        self.trees = extract_trees(page_tree, self.labels)
        self.items = extract_items(page_tree, self.trees, self.labels)
        self.item_fragments = np.where(
            self.items > 0, page_tree.index[self.items], -1)
