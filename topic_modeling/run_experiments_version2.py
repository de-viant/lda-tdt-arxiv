#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel
from gensim.matutils import Sparse2Corpus
from gensim.test.utils import datapath
from scipy import sparse
from itertools import product
from random import shuffle
from time import time

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)


def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return f'{h:>02.0f}:{m:>02.0f}:{s:>02.0f}'


text_path = Path('../data/clean_stop')
text_files = text_path.glob('*.txt')
docs = [f.read_text() for f in text_files]
cwd = Path().cwd()

min_dfs = [.001, .005, .01]
max_features = [None, 50000, 25000, 10000, 5000]

params = list(product(*[min_dfs, max_features]))
n = len(params)
shuffle(params)

cols = ['min_df', 'max_features', 'n_topics', 'coherence', 'std']
topic_coherence = []
experiment_path = Path('experiments_02')
corpus_path = experiment_path / 'corpora'
model_path = experiment_path / 'models'

start = time()
for i, (min_df, max_features) in enumerate(params, 1):
    print(min_df, max_features)
    key = f'{int(min_df*1000):d}_{max_features if max_features is not None else 0:d}'

    dtm_path = corpus_path / f'dtm_{key}.npz'
    token_path = corpus_path / f'tokens_{key}.csv'
    if not (dtm_path.exists() and token_path.exists()):

        vectorizer = CountVectorizer(min_df=min_df,
                                     max_features=max_features,
                                     binary=True)
        dtm = vectorizer.fit_transform(docs)
        sparse.save_npz(dtm_path, dtm)
        tokens = vectorizer.get_feature_names()
        pd.Series(tokens).to_csv(token_path, index=False)
    else:
        dtm = sparse.load_npz(dtm_path)
        tokens = pd.read_csv(token_path, header=None, squeeze=True).to_dict()

    id2word = pd.Series(tokens).to_dict()
    corpus = Sparse2Corpus(dtm, documents_columns=False)

    for n_topics in :
        print(n_topics, end=' ', flush=True)
        lda = LdaModel(corpus=corpus,
                       num_topics=n_topics,
                       id2word=id2word)
        model_file = datapath((model_path / f'{key}_{n_topics}').resolve())
        lda.save(model_file)
        coherence = [c[1] for c in lda.top_topics(corpus=corpus, coherence='u_mass')]
        topic_coherence.append([min_df, max_features, n_topics, np.mean(coherence), np.std(coherence)])

    elapsed = time() - start
    print(f'\nDone: {i/n:.2%} | Duration: {format_time(elapsed)} | To Go: {format_time(elapsed/i*(n-i))}\n')

    result = pd.DataFrame(topic_coherence, columns=cols).sort_values('coherence', ascending=False)

    print(result.sort_values('coherence', ascending=False).head(3))
    result.to_csv('experiments_02/coherence.csv', index=False)
