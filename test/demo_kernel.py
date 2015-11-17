import time
import sys
import codecs
import cgi

import scrapely.htmlpage as hp
import numpy as np

import aile.kernel
import aile.ptree


def annotate(page, labels, out_path="annotated.html"):
    match = aile.ptree.match_fragments(page.parsed_body)
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

        for i, (fragment, label) in enumerate(
                zip(page.parsed_body, labels)):
            if label >= 0:
                out.write('<span class="line" style="color:red">')
            else:
                out.write('<span class="line" style="color:black">')
            if isinstance(fragment, hp.HtmlTag):
                if fragment.tag_type == hp.HtmlTagType.CLOSE_TAG:
                    if match[i] >= 0 and indent > 0:
                        indent -= 1
                    write(u'{0:3d}|&lt;/{1}&gt;'.format(label, fragment.tag))
                else:
                    write(u'{0:3d}|&lt;{1}'.format(label, fragment.tag))
                    for k,v in fragment.attributes.iteritems():
                        out.write(u' {0}="{1}"'.format(k, v))
                    if fragment.tag_type == hp.HtmlTagType.UNPAIRED_TAG:
                        out.write('/')
                    out.write('&gt;')
                    if match[i] >= 0:
                        indent += 1
            else:
                write(u'{0:3d}|{1}'.format(
                    label,
                    cgi.escape(page.body[fragment.start:fragment.end].strip())))
            out.write('</span>\n')
        out.write("""
</pre>
</body>
</html>""")


if __name__ == '__main__':
    url = sys.argv[1]

    print 'Downloading URL...',
    t1 = time.clock()
    page = hp.url_to_page('https://news.ycombinator.com/')
    print 'done ({0}s)'.format(time.clock() - t1)

    print 'Extracting items...',
    t1 = time.clock()
    ie = aile.kernel.ItemExtract(aile.ptree.PageTree(page))
    print 'done ({0}s)'.format(time.clock() - t1)

    print 'Annotating HTML'
    labels = np.repeat(-1, len(ie.page_tree.page.parsed_body))
    for i in range(ie.item_fragments.shape[0]):
        for j in range(ie.item_fragments.shape[1]):
            labels[ie.item_fragments[i, j]] = j
    annotate(ie.page_tree.page, labels)
