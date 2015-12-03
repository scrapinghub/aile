import itertools

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
        self.n_nodes = len(self.nodes)
        reverse_index = np.repeat(-1, len(page.parsed_body))
        for i, idx in enumerate(self.index):
            reverse_index[idx] = i
        match = match_fragments(page.parsed_body)
        self.match = np.repeat(-1, self.n_nodes)
        self.parents = np.repeat(-1, self.n_nodes)
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

        self.n_children = np.zeros((self.n_nodes,), dtype=int)
        for p in self.parents:
            if p > -1:
                self.n_children[p] += 1
        self.max_childs = np.max(self.n_children)

        self.distance = np.ones((self.n_nodes, self.n_nodes), dtype=int)
        for i in range(self.n_nodes - 1, -1, -1):
            self.distance[i, i] = 0
            for a, b in itertools.combinations(self.children(i), 2):
                for j in range(a, max(a + 1, self.match[a])):
                    for k in range(b, max(b + 1, self.match[b])):
                        self.distance[j, k] = self.distance[j, a] + 2 + self.distance[b, k]
                        self.distance[k, j] = self.distance[j, k]

    def __len__(self):
        """Number of nodes in tree"""
        return len(self.index)

    def children(self, i):
        """An array with the indices of the direct children of node 'i'"""
        return i + 1 + np.flatnonzero(self.parents[i+1:max(i+1, self.match[i])] == i)

    def children_matrix(self, max_childs=None):
        """A matrix of shape (len(tree), max_childs) where row 'i' contains the
        children of node 'i'"""
        if max_childs is None:
            max_childs = self.max_childs
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
        """Siblings of node 'i'"""
        p = self.parents[i]
        if p != -1:
            return self.children(p)
        else:
            return np.flatnonzero(self.parents == -1)

    def prefix(self, i, stop_at=-1):
        """A path from 'i' going upwards up to 'stop_at'"""
        path = []
        p = i
        while p >= stop_at and p != -1:
            path.append(p)
            p = self.parents[p]
        return path

    def prefixes_at(self, i):
        """A list of paths going upwards that start at a descendant of 'i' and
        end at a 'i'"""
        paths = []
        for j in range(i, max(i+1, self.match[i])):
            paths.append(self.prefix(j, i))
        return paths

    def tree_size(self):
        """Return an array where the i-th entry is the size of subtree 'i'"""
        r = np.arange(len(self.match))
        s = r + 1
        return np.where(s > self.match, s, self.match) - r

    def fragment_index(self, tree_index):
        """Convert from tree node numbering to original fragment numbers"""
        return np.where(
            tree_index > 0, self.index[tree_index], -1)

    def is_descendant(self, parent, descendant):
        return descendant >= parent and \
            descendant < max(parent + 1, self.match[parent])

    def common_ascendant(self, nodes):
        s = set(range(self.n_nodes))
        for node in nodes:
            s &= set(self.prefix(node))
        return max(s)
