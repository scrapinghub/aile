import time

import scrapely.htmlpage as hp
import ete2

import aile.kernel as ker


def build_tree(ptree, labels=None):
    root = ete2.Tree(name='root')
    T = [ete2.Tree(name=(str(node) +'(' + str(i) + ')'))
         for i, node in enumerate(ptree.nodes)]
    if labels is not None:
        for t, lab in zip(T, labels):
            t.name += '{' + str(lab) + '}'
    for i, p in enumerate(ptree.parents):
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
#    page = hp.url_to_page('http://jobsearch.monster.co.uk/browse/?re=nv_gh_gnl1147_%2F')
    page = hp.url_to_page('https://news.ycombinator.com/')

    t1 = time.clock()
    page_tree = ker.PageTree(page)
    K = ker.kernel(page_tree)
    l = ker.cluster(page_tree, K)
    trees = ker.extract_trees(page_tree, l)
    items = ker.extract_items(page_tree, trees, l)
    print 'Total time: {0} seconds'.format(time.clock() - t1)
    t = build_tree(page_tree, labels=l)
    t.show()
