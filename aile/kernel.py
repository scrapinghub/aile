import collections
import itertools

import numpy as np
import sklearn.cluster
import networkx as nx

import _kernel as _ker
import dtw


def to_rows(d):
    """Make a square matrix with rows equal to 'd'.

    >>> print to_rows(np.array([1,2,3,4]))
    [[1 2 3 4]
     [1 2 3 4]
     [1 2 3 4]
     [1 2 3 4]]
     """
    return np.tile(d, (len(d), 1))


def to_cols(d):
    """Make a square matrix with columns equal to 'd'.

    >>> print ker.to_cols(np.array([1,2,3,4]))
    [[1 1 1 1]
     [2 2 2 2]
     [3 3 3 3]
     [4 4 4 4]]
    """
    return np.tile(d.reshape(len(d), -1), (1, len(d)))


def normalize_kernel(K):
    """New kernel with unit diagonal.

    K'[i, j] = K[i, j]/sqrt(K[i,i]*K[j,j])
    """
    d = np.diag(K).copy()
    d[d == 0] = 1.0
    return K/np.sqrt(to_rows(d)*to_cols(d))


def kernel_to_distance(K):
    """Build a distance matrix.

    From the dot product:
        |u - v|^2 = (u - v)(u - v) = u^2 + v^2 - 2uv
    """
    d = np.diag(K)
    return np.sqrt(to_rows(d) + to_cols(d) - 2*K)


def tree_size_distance(page_tree):
    """Build a distance matrix comparing subtree sizes.

    If T1 and T2 are trees and N1 and N2 the number of nodes within:
        |T1 - T2| = |N1 - N2|/(N1 + N2)
    Since:
        N1 >= 1
        N2 >= 1
    Then:
        0 <= |T1 - T2| < 1
    """
    s = page_tree.tree_size()
    a = to_cols(s).astype(float)
    b = to_rows(s).astype(float)
    return np.abs(a - b)/(a + b)


def cluster(page_tree, K, d1=1.0, d2=1.0, eps=0.76, min_samples=4):
    """Asign to each node in the tree a cluster label.

    It runs the DBSCAN algorithm with a distance matrix which is a
    linear combination:

        D = d1*D1 + d2*D2

    Where:
        D1 is the distance matrix from kernel K and
        D2 is the distance matrix from subtree sizes

    eps and min_samples are passed to DBSCAN as is

    Returns: for each node a label id. Label ID -1 means that the node
    is an outlier (it isn't part of any cluster).
    """
    clt = sklearn.cluster.DBSCAN(
        eps=eps, min_samples=min_samples, metric='precomputed')
    return clt.fit_predict(
        d1*kernel_to_distance(normalize_kernel(K)) +
        d2*tree_size_distance(page_tree))


def extract_label(ptree, labels, label_to_extract):
    """Extract all forests inside the labeled PageTree that are marked or have
    a sibling that is marked with label_to_extract.

    Returns: a list of tuples, where each tuple are the roots of the extracted
    subtrees.
    """
    roots = []
    i = 0
    while i < len(labels):
        children = ptree.children(i)
        if np.any(labels[children] == label_to_extract):
            first = None
            item = []
            for c in children:
                m = labels[c]
                if m != -1:
                    if first is None:
                        first = m
                    elif m == first:
                        roots.append(tuple(item))
                        item = []
                    item.append(c)
            if item:
                roots.append(item)
            i = ptree.match[i]
        else:
            i += 1
    return roots


def filter_labels(ptree, labels):
    """Assign children the labels of their parents, if any"""
    labels = labels.copy()
    for i, l in enumerate(labels):
        if l != -1:
            labels[i:max(i, ptree.match[i])] = l
    return labels


def score_labels(ptree, labels):
    """Assign an score for each label"""
    scores = collections.defaultdict(int)
    for i, l in enumerate(labels):
        if l != -1:
            scores[l] += max(0, ptree.match[i] - i + 1)
    return scores


def extract_trees(ptree, labels):
    """Extract the repeating trees.

    We cannot use the cluster labels as is because:
        1. If a tree repeats not only its root is assigned a label,
        most of its children too.
        2. A repating patter can be made of several distinct trees.

    The algorithm to extract the repeating trees goes as follows:
        1. Determine the label that covers most children on the page
        2. If a node with that label has siblings, extract the siblings too,
           even if they have other labels.
    """
    labels = filter_labels(ptree, labels)
    scores = score_labels(ptree, labels)
    max_s, max_l = max((s, l) for (l, s) in scores.iteritems())
    return extract_label(ptree, labels, max_l)


def path_distance(path_1, path_2):
    """Compute the prefix distance between the two paths.

    >>> p1 = [1, 0, 3, 4, 5, 6]
    >>> p2 = [1, 0, 2, 2, 2, 2, 2, 2]
    >>> print path_distance(p1, p2)
    6
    """
    d = max(len(path_1), len(path_2))
    for a, b in zip(path_1, path_2):
        if a != b:
            break
        d -= 1
    return d


def pairwise_path_distance(path_seq_1, path_seq_2):
    """Compute all pairwise distances between paths in path_seq_1 and
    path_seq_2"""
    N1 = len(path_seq_1)
    N2 = len(path_seq_2)
    D = np.zeros((N1, N2))
    for i in range(N1):
        q1 = path_seq_1[i]
        for j in range(N2):
            D[i, j] = path_distance(q1, path_seq_2[j])
    return D


def extract_path_seq_1(ptree, labels, forest):
    paths = []
    for root in forest:
        for path in ptree.prefixes_at(root):
            paths.append((path[0], labels[path].tolist()))
    return paths


def extract_path_seq(ptree, trees, labels):
    all_paths = []
    for tree in trees:
        paths = extract_path_seq_1(ptree, labels, tree)
        all_paths.append(paths)
    return all_paths


def find_cliques(G, min_size):
    """Find all cliques in G above a given size.

    If a node is part of a larger clique is deleted from the smaller ones.
    Returns:
        A dictionary mapping nodes to clique ID
    """
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


def match_graph(all_paths):
    """Build a graph where n1 and n2 share an edge if they have
    been matched using DTW"""
    G = nx.Graph()
    for path_set_1, path_set_2 in itertools.combinations(all_paths, 2):
        n1, p1 = zip(*path_set_1)
        n2, p2 = zip(*path_set_2)
        D = pairwise_path_distance(p1, p2)
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
            match_graph(extract_path_seq(ptree, trees, labels)),
            0.5*len(trees))
    )


class ItemExtract(object):
    def __init__(self, page_tree,
                 k_max_depth=2, k_decay=0.5,
                 c_d1=1.0, c_d2=1.0,
                 c_eps=0.76, c_min_samples=6):
        """Perform all extraction operations in sequence.

        Parameters:
            k_*: parameters to kernel computation
            c_*: parameters to clustering
        """
        self.page_tree = page_tree
        self.kernel = _ker.kernel(page_tree, max_depth=k_max_depth, decay=k_decay)
        self.labels = cluster(page_tree, self.kernel,
                              d1=c_d1, d2=c_d2, eps=c_eps, min_samples=c_min_samples)
        self.trees = extract_trees(page_tree, self.labels)
        self.items = extract_items(page_tree, self.trees, self.labels)
        self.item_fragments = np.where(
            self.items > 0, page_tree.index[self.items], -1)
