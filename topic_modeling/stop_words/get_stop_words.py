#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd

from pathlib import Path
import tempfile
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


def get_baseline():
    print('Spacy:', len(STOP_WORDS))
    sw_sk = set(stop_words.ENGLISH_STOP_WORDS)
    print('sklearn', len(sw_sk))

    sw = set(pd.read_csv('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words',
                         header=None,
                         squeeze=True).tolist())

    print('web', len(sw))

    all = STOP_WORDS.union(sw).union(sw_sk)
    print('all', len(all))
    pd.Series(sorted(list(all))).to_csv('baseline.csv', index=False)


def get_token_count():
    text_path = Path('../data/clean')
    text_files = text_path.glob('*.txt')
    token_count = Counter()
    for text_file in text_files:
        token_count.update(text_file.read_text().split())

    token_count = pd.Series(dict(token_count))
    token_count.to_csv('token_count.csv')


def combine_vocab():
    # sw = pd.read_csv('baseline.csv')
    # sw = sw.loc[sw['drop'].isnull(), ['token']].assign(stop=1)
    tokens = pd.read_csv('token_count.csv', na_values=[''], keep_default_na=False)
    print(tokens.info())

    english_words = set(pd.read_csv('..data/words_alpha.txt', squeeze=True).tolist())
    tokens['english'] = tokens.token.isin(english_words)
    tokens.to_csv('token_count.csv', index=False)
    print(tokens.english.value_counts())


def get_stop_words():
    tokens = pd.read_csv('token_count.csv', na_values=[''], keep_default_na=False)
    tokens['invalid'] = tokens.english.fillna(True).astype(int).sub(1).abs()
    print(tokens.invalid.value_counts())
    stop = tokens.loc[(tokens.invalid | tokens.stop), 'token']
    stop.to_csv('stop_words.csv', index=False)


def clean_docs():
    stop_words = set(pd.read_csv('stop_words.csv', header=None, squeeze=True))

    text_path = Path('../data/clean')
    clean_path = Path('../data/clean_stop')
    text_files = text_path.glob('*.txt')
    for i, text_file in enumerate(text_files, 1):
        if i % 1000 == 0:
            print(i, end=' ', flush=True)
        text = text_file.read_text().split()
        clean_text = [token for token in text if token not in stop_words]
        (clean_path / text_file.name).write_text(' '.join(clean_text))