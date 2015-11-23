# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import collections
import itertools

import numpy as np
cimport numpy as np
cimport cython

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


def similarity(ptree, max_items=4):
    all_classes = list({node.class_attr for node in ptree.nodes})
    class_index = {c: i for i, c in enumerate(all_classes)}
    class_map = np.array([class_index[node.class_attr] for node in ptree.nodes])
    N = len(all_classes)
    similarity = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            li = min(max_items, len(all_classes[i]))
            lj = min(max_items, len(all_classes[j]))
            lk = min(max_items, len(all_classes[i] & all_classes[j]))
            similarity[i, j] = (1.0 + lk)/(1.0 + li + lj - lk)
    return class_map, similarity


@cython.boundscheck(False)
cpdef build_counts(ptree, int max_depth=4, int max_childs=20):
    cdef int N = len(ptree)
    if max_childs is None:
        max_childs = N
    pairs = order_pairs(ptree.nodes)
    cdef np.ndarray[np.double_t, ndim=2] sim
    cdef np.ndarray[np.int_t, ndim=1] cmap
    cmap, sim = similarity(ptree)

    cdef np.ndarray[np.double_t, ndim=3] C = np.zeros((N, N, max_depth), dtype=float)
    cdef np.ndarray[np.double_t, ndim=3] S = np.zeros(
        (max_childs + 1, max_childs + 1, max_depth), dtype=float)
    S[0, :, :] = S[:, 0, :] = 1

    cdef int i1, i2, j1, j2, k1, k2
    cdef np.ndarray[np.int_t, ndim=2] children = ptree.children_matrix(max_childs)
    for i1, i2 in pairs:
        s = sim[cmap[i1], cmap[i2]]
        if children[i1, 0] == -1 and children[i2, 0] == -1:
            C[i2, i1, :] = C[i1, i2, :] = s
        else:
            for j1 in range(1, max_childs + 1):
                k1 = children[i1, j1 - 1]
                if k1 < 0:
                    break
                for j2 in range(1, max_childs + 1):
                    k2 = children[i2, j2 - 1]
                    if k2 < 0:
                        break
                    S[j1, j2,  :]  = S[j1 - 1, j2    , :  ] +\
                                     S[j1    , j2 - 1, :  ] -\
                                     S[j1 - 1, j2 - 1, :  ]
                    S[j1, j2, 1:] += S[j1 - 1, j2 - 1, :-1]*C[k1, k2, :-1]
            C[i2, i1, :] = C[i1, i2, :] = s*S[j1 - 1, j2 - 1, :]
    return C[:, :, max_depth - 1]


@cython.boundscheck(False)
cpdef kernel(ptree, counts=None, int max_depth=2, int max_childs=20, double decay=0.5):
    cdef np.ndarray[np.double_t, ndim=2] C
    if counts is None:
        C = build_counts(ptree, max_depth, max_childs)
    else:
        C = counts

    cdef np.ndarray[np.double_t, ndim=2] K = np.zeros((C.shape[0], C.shape[1]))
    cdef np.ndarray[np.double_t, ndim=2] A = C.copy()
    cdef np.ndarray[np.double_t, ndim=2] B = C.copy()
    cdef int N = K.shape[0]
    cdef int i, j, pi, pj, ri, rj
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
