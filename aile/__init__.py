import time

import scrapely

from . import slybot
from . import kernel
from . import ptree

def generate_slybot_project(url, path='slybot-project', verbose=False):
    def _print(s):
        if verbose:
            print s,

    _print('Downloading URL...')
    t1 = time.clock()
    page = scrapely.htmlpage.url_to_page(url)
    _print('done ({0}s)\n'.format(time.clock() - t1))

    _print('Extracting items...')
    t1 = time.clock()
    ie = kernel.ItemExtract(ptree.PageTree(page), separate_descendants=True)
    _print('done ({0}s)\n'.format(time.clock() - t1))

    _print('Generating slybot project...')
    t1 = time.clock()
    slybot.generate_slybot(ie, path)
    _print('done ({0}s)\n'.format(time.clock() - t1))

    return ie
