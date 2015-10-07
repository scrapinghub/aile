# Automatic Item List Extracation

This repository is a temporary container for experiments in automatic extraction of list and tables from web pages.
At some later point I will merge the surviving algorithms either in [scrapely](https://github.com/scrapy/scrapely) 
or [portia](https://github.com/scrapinghub/portia).

I document my ideas and algorithms descriptions at [readthedocs](http://aile.readthedocs.org/en/latest/).

The current approach is based on the HTML code of the page, treated as a stream of HTML tags as processed by 
[scrapely](https://github.com/scrapy/scrapely). 

We are trying to auto-detect repeating patterns in the tags, not necessarily made of of *li*, *tr* or *td* tags.
The problem of detecting repeating patterns in streams is known as *motif discovery* and most of the literature about it seems
to be published in the field of genetics.

Right now there are two algorithms: MEME and Profile HMM

## Installation	
	pip install -r requirements.txt
	python setup.py develop
	
## MEME
An initial and slow implementation of the [MEME](http://meme-suite.org/) algorithm, as described in the
following papers:

- Unsupervised learning of multiple motifs in biopolymers using
  expectation maximization.
  Bailey and Elkan, 1995

- Fitting a mixture model by expectation maximization to discover
  motifs in bipolymers.
  Bailey and Elkan, 1994

- The value of prior knowledge in discovering motifs with MEME.
  Bailey and Elkan, 1995

The algorithm is run without almost any particular information about HTML, just that candidate subsequences should have a 
balanced set of open/closing tags. In the future we could try to give more a priori probability to patterns starting with or containing list or table HTML tags.

### Running it    
    python meme.py
    
It will download and automatically extract the repating patterns of [Hacker News](https://news.ycombinator.com/). It actually works. The elements extracted are the articles. They have the form:

    <tr class='athing'>
      <td align="right" valign="top" class="title"><span class="rank">29.</span></td>

      <td>
        <center>
          <a id="up_10247910" href="vote?for=10247910&amp;dir=up&amp;goto=news" name=
          "up_10247910"></a>

          <div class="votearrow" title="upvote"></div>
        </center>
      </td>

      <td class="title"><a href=
      "https://www.textplain.net/blog/2015/moving-to-freebsd/">Moving to FreeBSD</a>
      <span class="sitebit comhead">(<a href="from?site=textplain.net"><span class=
      "sitestr">textplain.net</span></a>)</span></td>
    </tr>

    <tr>
      <td colspan="2"></td>

      <td class="subtext"><span class="score" id="score_10247910">35 points</span> by
      <a href="user?id=vezzy-fnord">vezzy-fnord</a> <a href="item?id=10247910">7 hours
      ago</a> | <a href="item?id=10247910">6 comments</a></td>
    </tr>

## Profile HMM
The main limitation of MEME is that detected sequences must have fixed length, that is, if one element of the motif is deleted or if noise is introduced between two motif elements it will fail to match.
This method is still work in progress and is being documented [here](http://aile.readthedocs.org/en/latest/notes.html)
