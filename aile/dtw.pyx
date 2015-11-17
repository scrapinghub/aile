import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
cpdef from_distance(np.ndarray[np.double_t, ndim=2] D):
    """Given a distance matrix compute the dynamic time warp distance.

    If the distance matrix 'D' is between sequences 's' and 't' then:
        1. D[i, j] = |s[i] - t[j]|
        2. DTW[i, j] represents the dynamic time warp distance between
           subsequences s[:i+1] and t[:j+1]
        3. DTW[-1, -1] is the dynamic time warp distance between s and t
    """
    cdef int m = D.shape[0]
    cdef int n = D.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] DTW = np.zeros((m + 1, n + 1))
    DTW[:, 0] = np.inf
    DTW[0, :] = np.inf
    DTW[0, 0] = 0
    cdef int i
    cdef int j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            DTW[i, j] = D[i - 1, j - 1] + min(
                DTW[i - 1, j    ],
                DTW[i    , j - 1],
                DTW[i - 1, j - 1])
    return DTW


@cython.boundscheck(False)
cpdef path(np.ndarray[np.double_t, ndim=2] DTW):
    """Given a DTW matrix backtrack to find the alignment between two sequences"""
    cdef int m = DTW.shape[0] - 1
    cdef int n = DTW.shape[1] - 1
    cdef int i = m - 1
    cdef int j = n - 1
    cdef np.ndarray[np.int_t, ndim=1] s = np.zeros((m,), dtype=int)
    cdef np.ndarray[np.int_t, ndim=1] t = np.zeros((n,), dtype=int)
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


@cython.boundscheck(False)
cpdef match(np.ndarray[np.int_t, ndim=1] s,
          np.ndarray[np.int_t, ndim=1] t,
          np.ndarray[np.double_t, ndim=2] D):
    """Given the alignments from two sequences find the match between elements.

    When aligning an element from one sequence can correspond to several elements
    of the other one. Matching resolves this ambiguity forcing unique pairings. If
    an element is unpaired then it as assigned -1.
    """
    s = s.copy()
    cdef int i, j, k, m
    cdef double d
    cdef int N = len(s)
    for i in range(N):
        j = s[i]
        m = k = i
        d = D[i, j]
        while k < N and s[k] == j:
            if D[k, j] < d:
                m = k
                d = D[k, j]
            k += 1
        k = i
        while k < N and s[k] == j:
            if k != m:
                s[k] = -1
            k += 1
    return s
