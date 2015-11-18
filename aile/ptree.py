import numpy as np
import scrapely.htmlpage as hp


def match_fragments(fragments, max_backtrack=20):
    """Find the closing fragment for every fragment.

    Returns: an array with as many elements as fragments. If the
    fragment has no closing pair then the array contains -1 at that position
    otherwise it contains the index of the closing pair.
    """
    match = np.repeat(-1, len(fragments))
    stack = []
    for i, fragment in enumerate(fragments):
        if isinstance(fragment, hp.HtmlTag):
            if fragment.tag_type == hp.HtmlTagType.OPEN_TAG:
                stack.append((i, fragment))
            elif (fragment.tag_type == hp.HtmlTagType.CLOSE_TAG):
                if max_backtrack is None:
                    max_j = len(stack)
                else:
                    max_j = min(max_backtrack, len(stack))
                for j in range(1, max_j + 1):
                    last_i, last_tag = stack[-j]
                    if (last_tag.tag == fragment.tag):
                        match[last_i] = i
                        match[i] = last_i
                        stack[-j:] = []
                        break
    return match


def is_tag(fragment):
    """Check if a fragment is also an HTML tag"""
    return isinstance(fragment, hp.HtmlTag)


def get_class(fragment):
    """Return a set with class attributes for a given fragment"""
    if is_tag(fragment):
        return frozenset((fragment.attributes.get('class') or '').split())
    else:
        return frozenset()


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
        return self.__repr__().encode('ascii', 'backslashreplace')

    def __repr__(self):
        s = unicode(self.tag)
        if self.class_attr:
            s += u'['
            s += u','.join(self.class_attr)
            s += u']'
        return s


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
        match = match_fragments(page.parsed_body)
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
