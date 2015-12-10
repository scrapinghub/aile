import time
import sys

import slyd.utils
import scrapely.htmlpage as hp
import numpy as np
import ete2
import matplotlib.pyplot as plt

import aile.kernel
import aile.ptree


def color_map(n_colors):
    cmap = plt.cm.Set3(np.linspace(0, 1, n_colors))
    cmap = np.round(cmap[:,:-1]*255).astype(int)
    def to_hex(c):
        return hex(c)[2:-1]
    return ['#' + ''.join(map(to_hex, row)) for row in cmap]


def draw_tree(ptree, labels=None):
    root = ete2.Tree(name='root')
    T = [ete2.Tree(name=(str(node) + '[' + str(i) + ']'))
         for i, node in enumerate(ptree.nodes)]
    if labels is not None:
        for t, lab in zip(T, labels):
            t.name += '{' + str(lab) + '}'
    for i, p in enumerate(ptree.parents):
        if p > 0:
            T[p].add_child(T[i])
        else:
            root.add_child(T[i])
    cmap = color_map(max(labels) + 2)
    for t, l in zip(T, labels):
        ns = ete2.NodeStyle()
        ns['bgcolor'] = cmap[l]
        t.set_style(ns)
        if not t.is_leaf():
            t.add_face(ete2.TextFace(t.name), column=0, position='branch-top')
    root.show()


if __name__ == '__main__':
    url = sys.argv[1]

    print 'Downloading URL...',
    t1 = time.clock()
    page = hp.url_to_page(url)
    print 'done ({0}s)'.format(time.clock() - t1)

    print 'Extracting items...',
    t1 = time.clock()
    ie = aile.kernel.ItemExtract(aile.ptree.PageTree(page))
    print 'done ({0}s)'.format(time.clock() - t1)

    print 'Drawing HTML tree'
    draw_tree(ie.page_tree, ie.labels)
