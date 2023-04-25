#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import argparse
import random
from _datetime import datetime
from pathlib import Path
from time import mktime, sleep
from shutil import copyfile

import feedparser
import numpy as np
import pandas as pd
import requests

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

id_map = pd.read_csv('meta_data.csv.gz')
id_map.id = id_map.id.apply(lambda x: x.split('v')[0])
id_map = id_map.set_index('id').title.to_dict()
# print(id_map)

stored = Path('/drive/data/NLP/arxiv/pdf').glob('**/*.pdf')
for f in stored:
    if f.stem in id_map.keys():
        dst = 'selected_papers/' + id_map[f.stem] + '.pdf'
        copyfile(str(f), dst)

exit()


def download_pdf():
    fails = 0
    path = Path('papers')
    done = [str(f).split('/')[-1] for f in path.glob('*.pdf')]
    redo = []
    df = pd.read_csv('meta_data.csv')

    for i, (id, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            print(i, end=' ', flush=True)
        url = row.pdf_url.replace('/abs/', '/pdf/') + '.pdf'
        file_name = row.title + '.pdf'
        if file_name in done:
            continue
        try:
            r = requests.get(url)
            if 'PDF unavailable for' in r.text:
                redo.append(file_name)
                continue
            (path / file_name).write_bytes(r.content)
            sleep(2)
        except:
            if fails < 5:
                fails += 1
                print(f'Fail: {fails}')gst

                sleep(1200)
            else:
                pd.DataFrame({'errors': redo}).to_csv('redo.csv', index=False)
                exit()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=int, default=100, help='results per page')
    parser.add_argument('--wait', type=float, default=5.0, help='time between requests')
    return parser.parse_args()


def encode_feedparser_dict(d):
    """
    helper function to get rid of feedparser bs with a deep copy.
    I hate when libs wrap simple things in their own classes.
    """
    if isinstance(d, feedparser.FeedParserDict) or isinstance(d, dict):
        j = {}
        for k in d.keys():
            j[k] = encode_feedparser_dict(d[k])
        return j
    elif isinstance(d, list):
        l = []
        for k in d:
            l.append(encode_feedparser_dict(k))
        return l
    else:
        return d


def parse_arxiv_url(url):
    """
    examples is http://arxiv.org/abs/1512.08756v2
    we want to extract the raw id and the version
    """
    ix = url.rfind('/')
    idversion = url[ix + 1:]  # extract just the id (and the version)
    parts = idversion.split('v')
    assert len(parts) == 2, 'error parsing url ' + url
    return parts[0], int(parts[1])


def save_articles(df):
    path = Path('/home/ubuntu')
    (df.T
     .reset_index()
     .rename(columns={'index': 'id'})
     .drop_duplicates('id')
     .to_csv(path / f'all_articles.csv', mode='a', header=False, index=False))


def download(start=0, end=100000, max_results=10, wait=10, done=None):
    no_results = 0
    base_url = 'http://export.arxiv.org/api/query?'
    categories = ['cat:cs.CV', 'cat:cs.AI', 'cat:cs.LG', 'cat:cs.CL', 'cat:cs.NE', 'cat:stat.ML']
    params = {'search_query': '+OR+'.join(categories),
              'sortBy'      : 'lastUpdatedDate',
              'max_results' : max_results}

    articles = pd.DataFrame()
    for s in range(start, end, max_results):
        print(s, end=' ', flush=True)
        params.update({'start': s})
        param_str = "&".join(f'{k}={v}' for k, v in params.items())
        response = requests.get(url=base_url, params=param_str)
        parsed = feedparser.parse(response.text)

        if len(parsed.entries) == 0:
            print('\nReceived no results from arxiv.')
            no_results += 1
            if no_results < 5:
                sleep(60)
            else:
                save_articles(articles)
                break

        for entry in parsed.entries:
            entry = encode_feedparser_dict(entry)
            id = entry['id'].split('/')[-1]
            if id in done:
                continue
            r = dict(
                    summary=' '.join([t.strip() for t in entry['summary'].replace('\n', ' ').split()]),
                    tags=','.join([t['term'] for t in entry['tags']]),
                    category=entry['arxiv_primary_category']['term'],
                    affiliation=entry.get('arxiv_affiliation'),
                    pdf_url=entry['link'],
                    author=entry['author'],
                    updated=datetime.fromtimestamp(mktime(entry['updated_parsed'])),
                    published=datetime.fromtimestamp(mktime(entry['published_parsed'])),
                    title=entry['title'].replace('\n', ' ')
            )
            articles[id] = pd.Series(r)
        sleep(wait + random.uniform(0, 3))

    print(articles.T.info(null_counts=True))
    save_articles(articles)


if __name__ == '__main__':
    download_pdf()
    # args = parse_args()
    # done = pd.read_csv(path / 'all_articles.csv').drop_duplicates('id')
    # download(start=len(done), max_results=args.results, wait=args.wait, done=done.id.tolist())
