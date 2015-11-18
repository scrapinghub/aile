# Automatic Item List Extracation

This repository is a temporary container for experiments in automatic extraction of list and tables from web pages.
At some later point I will merge the surviving algorithms either in [scrapely](https://github.com/scrapy/scrapely)
or [portia](https://github.com/scrapinghub/portia).

I document my ideas and algorithms descriptions at [readthedocs](http://aile.readthedocs.org/en/latest/).

The current approach is based on the HTML code of the page, treated as a stream of HTML tags as processed by
[scrapely](https://github.com/scrapy/scrapely). An alternative approach would be to use also the web page
rendering information ([this script](https://github.com/plafl/aile/blob/master/misc/visual.py) renders a tree
of bounding boxes for each element).

## Installation
	pip install -r requirements.txt
	python setup.py develop

## Running
If you want to have a feeling of how it works there are two demo scripts included in the repo.

- demo1.py
  Will annotate the HTML code of a web page, marking as red the lines that form part of the repeating item
  and with a prefix number the field number inside the item. The output is written in the file 'annotated.html'.

      python demo1.py https://news.ycombinator.com

  ![annotated HTML](https://github.com/plafl/aile/blob/master/misc/demo1_img.png)

- demo2.py
  Will label, color and draw the HTML tree so that repeating elements are easy to see. The output is interactive
  (requires PyQt4).

      python demo2.py https://news.ycombinator.com

  ![annotated tree](https://github.com/plafl/aile/blob/master/misc/demo2_img.png)

## Algorithms

We are trying to auto-detect repeating patterns in the tags, not necessarily made of of *li*, *tr* or *td* tags.

### Clustering trees with a measure of similarity
The idea is to compute the distance between all subtrees in the web page and run a clustering algorithm with this distance matrix.
For a web page of size N this can be achieved in time O(N^2). The current algorithm actually computes a kernel and from the kernel
computes the distance. The algorithm is based on:

    Kernels for semi-structured data
    Hisashi Kashima, Teruo Koyanagi

Once we compute the distance between all subtrees of the web page DBSCAN clustering is run using the distance matrix.
The result is massaged a little more until you get the result.

### Markov models
The problem of detecting repeating patterns in streams is known as *motif discovery* and most of the literature about it seems
to be published in the field of genetics. Inspired from this there is [a branch](https://github.com/plafl/aile/tree/markov_model)
(MEME and Profile HMM algorithms).

The Markov model approach has the following problems right now:

- Requires several web pages for training, depending on the web page type
- Training is performed using EM algorithm which requires several attempts until a good optimum is achieved
- The number of hidden states is hard to determine. There are some heuristics applied that work partially

These problems are not unsurmountable (I think) but require a lot of work:

- Precision could be improved using [conditional random fields](https://en.wikipedia.org/wiki/Conditional_random_field).
  These could alleviate the need for data.
- Training can run greatly in parallel. This is actually already done using [joblib](https://pythonhosted.org/joblib/parallel.html)
  in a single PC but it could be further improved using a cluster of computers
- There are some papers about hidden state merging/splitting and even an
  [infinite number of states](http://machinelearning.wustl.edu/mlpapers/paper_files/nips02-AA01.pdf)
