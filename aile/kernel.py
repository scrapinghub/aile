import collections
import itertools

import ete2
import numpy as np
import scipy.sparse as sparse
import scrapely.htmlpage as hp
import sklearn.cluster

import page_extractor as pe


def expand_class(fragments):
    r = []
    for fragment in fragments:
        if isinstance(fragment, hp.HtmlTag):
            class_attr = fragment.attributes.get('class')
            if class_attr:
                new_attr = dict(fragment.attributes)
                del new_attr['class']
                r.append(
                    hp.HtmlTag(
                        hp.HtmlTagType.OPEN_TAG,
                        fragment.tag,
                        fragment.attributes,
                        fragment.start,
                        fragment.end))
                for c in class_attr.split():
                    r.append(
                        hp.HtmlTag(
                            hp.HtmlTagType.UNPAIRED_TAG,
                            '[class]' + c,
                            dict(),
                            fragment.start,
                            fragment.end))
                if fragment.tag_type == hp.HtmlTagType.UNPAIRED_TAG:
                    r.append(
                        hp.HtmlTag(
                            hp.HtmlTagType.CLOSE_TAG,
                            fragment.tag,
                            dict(),
                            fragment.start,
                            fragment.end))
            else:
                r.append(fragment)
        else:
            r.append(fragment)
    return r


def filter_empty_text(page):
    """Return a list of fragments from page where empty text has been deleted"""
    return [fragment
            for fragment in page.parsed_body
            if (not fragment.is_text_content or
                page.body[fragment.start:fragment.end].strip())]


def jaccard_index(s1, s2):
    """Compute Jaccard index between two sets"""
    if s1 or s2:
        I = float(len(s1 & s2))
        return I / (len(s1) + len(s2) - I)
    else:
        return 1.0


def is_tag(fragment):
    """Check if a fragment is also an HTML tag"""
    return isinstance(fragment, hp.HtmlTag)


def get_class(fragment):
    """Return a set with class attributes for a given fragment"""
    if is_tag(fragment):
        return set((fragment.attributes.get('class') or '').split())
    else:
        return set()


def class_similarity(f1, f2):
    return jaccard_index(get_class(f1), get_class(f2))


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


def children(fragments, match, i):
    """Children of the i-th fragment"""
    return [i + 1 + j for j, fragment in get_nodes(fragments[i+1: max(match[i], i+1)])]


def is_ascendant(match, i_child, i_ascendant):
    """True if the second fragment is an ascendant of the first"""
    return i_child >= i_ascendant and i_child <= match[i_ascendant]


def build_counts(fragments, max_depth=4, sim=class_similarity, max_childs=50):
    match = pe.match_fragments(fragments)
    parents = pe.build_tree(match)
    pairs = order_pairs(fragments)
    N = len(fragments)
    C = np.zeros((N, N, max_depth), dtype=float)
    S = np.zeros((N + 1, N + 1, max_depth), dtype=float)
    S[0, :, :] = S[:, 0, :] = 1
    for i1, i2 in pairs:
        ch1 = children(fragments, match, i1)
        ch2 = children(fragments, match, i2)
        if not ch1 and not ch2:
            C[i2, i1, :] = C[i1, i2, :] = sim(fragments[i1], fragments[i2])
        else:
            nc1 = len(ch1)
            nc2 = len(ch2)
            for j1 in range(1, nc1 + 1):
                for j2 in range(1, nc2 + 1):
                    S[j1, j2,  :]  = S[j1 - 1, j2    , :  ] +\
                                     S[j1    , j2 - 1, :  ] -\
                                     S[j1 - 1, j2 - 1, :  ]
                    S[j1, j2, 1:] += S[j1 - 1, j2 - 1, :-1]*C[ch1[j1 - 1], ch2[j2 - 1], :-1]
            C[i2, i1, :] = C[i1, i2, :] = \
                        sim(fragments[i1], fragments[i2])*S[nc1, nc2, :]
    return C[:, :, max_depth - 1]


def kernel(counts, parents):
    K = np.zeros(counts.shape)
    N = K.shape[0]
    for i in range(N - 1, -1, -1):
        for j in range(N - 1, -1, -1):
            K[i, j] += counts[i, j]
            pi = parents[i]
            pj = parents[j]
            if pi > 0:
                K[pi, j] += K[i, j]
            if pj > 0:
                K[i, pj] += K[i, j]
    return K


def dist_matrix(K):
    d = np.diag(K)
    N = len(d)
    return np.sqrt(
        np.tile(d, (N, 1)) + np.tile(d.reshape(N, -1), (1, N)) - 2*K)


def cluster(match, D):
    for i in range(len(match)):
        j = max(i+1, match[i]+1)
        D[i, i+1:j] = np.inf
        D[i+1:j, i] = np.inf
    clt = sklearn.cluster.DBSCAN(eps=1.0, min_samples=4, metric='precomputed')
    lab = clt.fit_predict(D)
    grp, cnt = np.unique(lab, return_counts=True)
    


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


def build_tree(fragments, parents):
    root = ete2.Tree(name='root')
    T = [ete2.Tree(name=(fragment_name(f)+'(' + str(i) + ')'))
         for i, f in enumerate(fragments)]
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
    page = hp.url_to_page('https://patchofland.com/')

    fragments = filter_empty_text(page)
    C = build_counts(fragments)
    match = pe.match_fragments(fragments)
    parents = pe.build_tree(match)
    t = build_tree(fragments, parents)
