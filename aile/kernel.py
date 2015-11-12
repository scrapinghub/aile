import collections
import itertools

import ete2
import numpy as np
import scrapely.htmlpage as hp
import sklearn.cluster

import aile.page_extractor as pe


def filter_empty_text(page):
    """Return a list of fragments from page where empty text has been deleted"""
    return [fragment
            for fragment in page.parsed_body
            if (not fragment.is_text_content or
                page.body[fragment.start:fragment.end].strip())]


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
        s = set((fragment.attributes.get('class') or '').split())
        s.add(fragment.tag)
        return s
    else:
        s = set()
        if fragment.is_text_content:
            s.add('[T]')
        return s


def class_similarity(f1, f2, no_class=1.0):
    return jaccard_index(get_class(f1), get_class(f2), no_class)


def match_class(f1, f2, threshold=0.5):
    """Check if two fragments agree on their class attributes"""
    return class_similarity >= threshold


def tag_equal(f1, f2):
    """Check if two tags are equal"""
    return (f1.tag_type == f2.tag_type and f1.tag == f2.tag and
            match_class(f1, f2))


def fragment_equal(f1, f2):
    """Check if two fragments are equal"""
    return ((is_tag(f1) and is_tag(f2) and tag_equal(f1, f2)) or
            (f1.is_text_content and f2.is_text_content))


def fragment_to_node(fragment):
    """Convert a fragment to a node inside a tree where we are going
    to compute the kernel"""
    if fragment.is_text_content:
        return '[T]'
    elif (isinstance(fragment, hp.HtmlTag) and
          fragment.tag_type != hp.HtmlTagType.CLOSE_TAG):
        return fragment.tag
    return None


def get_nodes(fragments):
    for i, fragment in enumerate(fragments):
        node = fragment_to_node(fragment)
        if node:
            yield (i, node)


def order_pairs(fragments):
    """Given a list of fragments return pairs of equal nodes ordered in
    such a way that if pair_1 comes before pair_2 then no element of pair_2 is
    a descendant of pair_1"""
    grouped = collections.defaultdict(list)
    for i, name in get_nodes(fragments):
        grouped[name].append(i)
    return sorted(
        [pair
         for name, indices in grouped.iteritems()
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


def children(fragments, match, parents, i):
    """Children of the i-th fragment"""
    return [k for k in [i + 1 + j for j, fragment in get_nodes(fragments[i+1: max(match[i], i+1)])]
            if parents[k] == i]


def is_ascendant(match, i_child, i_ascendant):
    """True if the second fragment is an ascendant of the first"""
    return i_child >= i_ascendant and i_child <= match[i_ascendant]


def build_counts(fragments, match, parents,
                 max_depth=4, sim=class_similarity, max_childs=20):
    N = len(fragments)
    if max_childs is None:
        max_childs = N
    pairs = order_pairs(fragments)
    C = np.zeros((N, N, max_depth), dtype=float)
    S = np.zeros((max_childs + 1, max_childs + 1, max_depth), dtype=float)
    S[0, :, :] = S[:, 0, :] = 1
    for i1, i2 in pairs:
        ch1 = children(fragments, match, parents, i1)
        ch2 = children(fragments, match, parents, i2)
        if not ch1 and not ch2:
            C[i2, i1, :] = C[i1, i2, :] = sim(fragments[i1], fragments[i2])
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
                        sim(fragments[i1], fragments[i2])*S[nc1, nc2, :]
    return C[:, :, max_depth - 1]


def kernel(fragments, match=None, parents=None, counts=None,
           max_depth=2, sim=class_similarity, max_childs=20, decay=0.1):
    if match is None:
        match = pe.match_fragments(fragments)
    if parents is None:
        parents = pe.build_tree(match)
    if counts is None:
        C = build_counts(fragments, match, parents, max_depth, sim, max_childs)
    else:
        C = counts
    K = np.zeros(C.shape)
    N = K.shape[0]

    A = C.copy()
    B = C.copy()
    for i in range(N - 1, -1, -1):
        pi = parents[i]
        for j in range(N - 1, -1, -1):
            pj = parents[j]
            if pi > 0:
                A[pi, j] += decay*A[i, j]
            if pj > 0:
                B[i, pj] += decay*B[i, j]
    for i in range(N - 1, -1, -1):
        pi = parents[i]
        for j in range(N - 1, -1, -1):
            ri = max(match[i], i) + 1
            rj = max(match[j], j) + 1
            K[i, j] += A[i, j] + B[i, j] - C[i, j]
            pj = parents[j]
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


def cluster(fragments, match, parents, K):
    D = kernel_to_distance(normalize_kernel(K))
    clt = sklearn.cluster.DBSCAN(eps=0.76, min_samples=8, metric='precomputed')
    lab = clt.fit_predict(D)
    grp = collections.defaultdict(list)
    for i, l in enumerate(lab):
        grp[l].append(i)
    grp = {k: np.array(v) for k, v in grp.iteritems()}
    scores = {k: sum(max(0, match[i] - i) for i in v)
              for k, v in grp.iteritems()}
    return lab, grp, scores


def fragment_name(fragment):
    if fragment.is_text_content:
        return '[T]'
    elif isinstance(fragment, hp.HtmlTag):
        class_attr = fragment.attributes.get('class')
        if class_attr:
            return fragment.tag + '[' + class_attr + ']'
        else:
            return fragment.tag
    else:
        return ''


def build_tree(fragments, parents, labels=None):
    root = ete2.Tree(name='root')
    T = [ete2.Tree(name=(fragment_name(f)+'(' + str(i) + ')'))
         for i, f in enumerate(fragments)]
    if labels is not None:
        for t, lab in zip(T, labels):
            t.name += 'lab=' + str(lab)
    for i, (f, p) in enumerate(zip(fragments, parents)):
        if isinstance(f, hp.HtmlTag) and f.tag_type == hp.HtmlTagType.CLOSE_TAG:
            continue
        if p > 0:
            T[p].add_child(T[i])
        else:
            root.add_child(T[i])
    for t in root.traverse():
        if not t.is_leaf():
            t.add_face(ete2.TextFace(t.name), column=0, position='branch-top')
    return root


if __name__ == '__main__':
#    page = hp.url_to_page('http://www.ebay.com/sch/Car-and-Truck-Tires/66471/bn_584423/i.html')
#    page = hp.url_to_page('https://patchofland.com/investments.html')
    page = hp.url_to_page('http://jobsearch.monster.co.uk/browse/?re=nv_gh_gnl1147_%2F')
    fragments = filter_empty_text(page)
    match = pe.match_fragments(fragments)
    parents = pe.build_tree(match)

    K = kernel(fragments, match, parents)
    l, c, s = cluster(fragments, match, parents, K)
    t = build_tree(fragments, parents, labels=l)
