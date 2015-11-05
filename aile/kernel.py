import collections
import itertools

import numpy as np
import scipy.sparse as sparse
import scrapely.htmlpage as hp

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
    

def order_pairs(fragments):
    def convert(fragments):
        for i, fragment in enumerate(fragments):
            if fragment.is_text_content:
                yield (i, '[T]')
            elif (isinstance(fragment, hp.HtmlTag) and
                  fragment.tag_type != hp.HtmlTagType.CLOSE_TAG):
                yield (i, fragment.tag)

    grouped = collections.defaultdict(list)
    for i, fragment in convert(fragments):
        grouped[fragment].append(i)
    return sorted(
        [pair
         for fragment, indices in grouped.iteritems()
         for pair in itertools.combinations_with_replacement(sorted(indices), 2)],
         key=lambda pair: min(pair),
         reverse=True
        )


def is_ascendant(parents, i_child, i_ascendant):
    j = i_child
    while j != -1:
        j = parents[j]
        if j == i_ascendant:
            return True
    return False


def build_counts(fragments, pairs, decay=1.0,
                 use_sparse=False, filter_ascendants=True, max_depth=4):
    match = pe.match_fragments(fragments)
    parents = pe.build_tree(match)
    N = len(fragments)
    if use_sparse:
        C = sparse.dok_matrix((N, N, max_depth), dtype=float)
    else:
        C = np.zeros((N, N, max_depth), dtype=float) - 1e6
    for i1, i2 in pairs:
        if not filter_ascendants or not is_ascendant(parents, i2, i1):
            for d in range(max_depth):
                # C[i1, i2, d] = decay            
                C[i1, i2, d] = np.log(decay) 
    for i1, i2 in pairs:
        p1 = parents[i1]
        p2 = parents[i2]
        if p1>0 and p2>0:
            if p2 < p1:
                p1, p2 = p2, p1
            for d in range(1, max_depth):
                # C[p1, p2, d] *= 1.0 + C[i1, i2, d - 1]
                C[p1, p2, d] += np.logaddexp(0, C[i1, i2, d-1])
    return C[:, :, max_depth - 1]
        
    
if __name__ == '__main__':
    pass
