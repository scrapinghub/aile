import os
import sys
import codecs
import urllib
import cgi

import numpy as np
import scrapely.htmlpage as hp

from aile.phmm import ProfileHMM
import aile.page_extractor as pe

def phmm_cmp(W, Z1, Z2):
    return ((Z1 >= W) != (Z2 >= W)).mean()


def demo1():
    phmm_true = ProfileHMM(
        f=np.array([
            [0.2, 0.3, 0.2, 0.3],
            [0.9, 0.1, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2]]),
        t=np.array([
            0.05, 0.9, 0.1, 0.05, 0.85, 0.1])
    )

    X, Z = phmm_true.generate(5000)
    phmm = ProfileHMM.fit(X, 3)

    print "True model 't' parameters", phmm_true.t
    print " Estimated 't' paramaters", phmm.t

    z, logP = phmm.viterbi(X)
    print 'Error finding motifs (% mismatch):', phmm_cmp(phmm.W, Z, z)*100


def train_test(pattern, start, end):
    return ([pattern.format(i) for i in range(start, end + 1)],
            pattern.format(end + 1))


def train_test_1(n_train=2):
    return ('hn',
            train_test('https://news.ycombinator.com/news?p={0}', 1, n_train + 1))


def train_test_2(n_train=1):
    return ('patchofland',
            train_test('https://patchofland.com/investments/page/{0}.html', 1, n_train + 1))


def train_test_3(n_train=6):
    return ('ebay',
            train_test('http://www.ebay.com/sch/Tires-/66471/i.html?_pgn={0}', 1, n_train + 1))


def train_test_4(n_train=6):
    return ('monster',
            train_test('http://jobsearch.monster.co.uk/browse/?pg={0}&re=nv_gh_gnl1147_%2F', 1, n_train + 1))


def train_test_5(n_train=3):
    pattern = 'http://lambda-the-ultimate.org/node?from={0}'
    return ('lambda', ([pattern.format(i) for i in range(0, n_train*10, 10)],
                       pattern.format(n_train*10)))


def train_test_6(n_train=3):
    return ('arstechnica',
            train_test('http://arstechnica.com/page/{0}/', 1, n_train + 1))


def download(train_test):
    root, (train, test) = train_test
    train_download = ['{0}-{1}.html'.format(root, i) for i in range(len(train))]
    for url, local in zip(train, train_download):
        if not os.path.exists(local):
            print '{0} -> {1}'.format(url, local)
            urllib.urlretrieve(url, local)
        else:
            print 'Using local version {0} of URL {1}'.format(local,url)
    test_download = '{0}-{1}.html'.format(root, len(train) + 1)
    if not os.path.exists(test_download):
        urllib.urlretrieve(test, test_download)
    return (map(make_local_url, train_download),
            make_local_url(test_download))


def make_local_url(path):
    return 'file:///' + os.path.abspath(path)


def annotate(fit_result, page_sequence, out_path="annotated.html"):
    X = np.array(map(fit_result.code_book.code, page_sequence.tags))
    Z, logP = fit_result.model.viterbi(X)
    match = pe.match_fragments(page_sequence.fragments)

    with codecs.open(out_path, 'w', encoding='utf-8') as out:
        out.write("""
<!DOCTYPE html>
<html lang="en-US">
<head>
<style>
    pre {
      counter-reset: code;
      padding-left: 30px;
    }

    .line {
      counter-increment: code;
    }

    .line:before {
      content: counter(code);
      float: left;
      margin-left: -30px;
      width: 25px;
      text-align: right;
    }
</style>
</head>
<body>
<pre>
""")
        indent = 0
        def write(s):
            out.write(indent*'    ')
            out.write(s)

        for i, (fragment, z) in enumerate(zip(page_sequence.fragments, Z)):
            if z >= fit_result.model.W:
                state = z - fit_result.model.W
            else:
                state = -1
            if state >= 0:
                out.write('<span class="line" style="color:red">')
            else:
                out.write('<span class="line" style="color:black">')
            if isinstance(fragment, hp.HtmlTag):
                if fragment.tag_type == hp.HtmlTagType.CLOSE_TAG:
                    if match[i] >= 0 and indent > 0:
                        indent -= 1
                    write(u'{0:3d}|&lt;/{1}&gt;'.format(state, fragment.tag))
                else:
                    write(u'{0:3d}|&lt;{1}'.format(state, fragment.tag))
                    for k,v in fragment.attributes.iteritems():
                        out.write(u' {0}="{1}"'.format(k, v))
                    if fragment.tag_type == hp.HtmlTagType.UNPAIRED_TAG:
                        out.write('/')
                    out.write('&gt;')
                    if match[i] >= 0:
                        indent += 1
            else:
                write(u'{0:3d}|{1}'.format(state,
                    cgi.escape(page_sequence.body_segment(i).strip())))
            out.write('</span>\n')
        out.write("""
</pre>
</body>
</html>""")


def demo2(train_test, out='demo'):
    train_urls, test_url = download(train_test)
    train = pe.PageSequence([hp.url_to_page(url) for url in train_urls])
    fit_result = pe.fit_model(train)
    for fr in fit_result:
        outf = codecs.open(
            '{0}-{1}.html'.format(out, fr.model.W), 'w', encoding='utf-8')
        fr.items.to_html(outf)
        annotate(fr, train, '{0}-annotated-{1}.html'.format(out, fr.model.W))
        print fr.model.W, fr.logP, pe.items_score(fr.items), fr.model.motif_entropy

    return train, fit_result

if __name__ == '__main__':
    tests = [
        train_test_1,
        train_test_2,
        train_test_3,
        train_test_4,
        train_test_5,
        train_test_6
    ]

    n_test = int(sys.argv[1])
    train, fr = demo2(tests[n_test-1](), out='demo-{0}'.format(n_test))
