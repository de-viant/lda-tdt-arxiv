#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.test.utils import datapath
from scipy import sparse
from itertools import product
from random import shuffle
from time import time
import re

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)


def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return f'{h:>02.0f}:{m:>02.0f}:{s:>02.0f}'


# experiment setup
cols = ['vocab_size', 'test_vocab', 'max_features', 'n_topics', 'perplexity', 'u_mass']

topic_coherence = []
experiment_path = Path('experiments_06')
corpus_path = experiment_path / 'corpora'
model_path = experiment_path / 'models'
for path in [corpus_path, model_path]:
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)

# get text files
text_path = Path('../data/clean_stop')
text_files = text_path.glob('*.txt')
docs = [f.read_text() for f in text_files]

train_docs, test_docs = train_test_split(docs, test_size=.1)
# texts = [re.findall(r'(?u)\b\w\w+\b', t) for t in train_docs]

# experiment params
min_df = .001
max_features = [10000, 25000]
# params = list(product(*[min_dfs, max_features]))
n = len(max_features)
# shuffle(params)

start = time()
for i, max_features in enumerate(max_features, 1):
    print(max_features)
    key = f'{max_features if max_features is not None else 0:d}'

    dtm_path = corpus_path / f'dtm_{key}.npz'
    token_path = corpus_path / f'tokens_{key}.csv'

    vectorizer = CountVectorizer(min_df=min_df,
                                 max_features=max_features,
                                 binary=True)
    train_dtm = vectorizer.fit_transform(train_docs)
    train_corpus = Sparse2Corpus(train_dtm, documents_columns=False)
    train_tokens = vectorizer.get_feature_names()

    test_dtm = vectorizer.transform(test_docs)
    test_corpus = Sparse2Corpus(test_dtm, documents_columns=False)
    test_vocab = test_dtm.count_nonzero()

    dtm = vectorizer.fit_transform(docs)
    sparse.save_npz(dtm_path, dtm)
    tokens = vectorizer.get_feature_names()
    vocab_size = len(tokens)
    pd.Series(tokens).to_csv(token_path, index=False)

    id2word = pd.Series(tokens).to_dict()
    corpus = Sparse2Corpus(dtm, documents_columns=False)

    # dictionary = Dictionary.from_corpus(corpus=train_corpus, id2word=id2word)

    # for n_topics in [3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 100]:
    for n_topics in [5, 10, 15, 20, 30]:
        print(n_topics, end=' ', flush=True)
        lda = LdaModel(corpus=corpus,
                       num_topics=n_topics,
                       id2word=id2word)

        doc_topics = pd.DataFrame()
        for i, topics in enumerate(lda.get_document_topics(corpus)):
            doc_topics = pd.concat([doc_topics, pd.DataFrame(topics, columns=['topic', 'value']).assign(doc=i)])
        doc_topics.to_csv(model_path / f'doc_topics_{key}_{n_topics}.csv', index=False)

        model_file = datapath((model_path / f'{key}_{n_topics}').resolve())
        lda.save(model_file)
        train_lda = LdaModel(corpus=train_corpus,
                             num_topics=n_topics,
                             id2word=pd.Series(train_tokens).to_dict())

        # see https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel.log_perplexity
        test_perplexity = 2 ** (-train_lda.log_perplexity(test_corpus))

        # https://markroxor.github.io/gensim/static/notebooks/topic_coherence_tutorial.html
        u_mass = np.mean([c[1] for c in lda.top_topics(corpus=corpus, coherence='u_mass', topn=n_topics)])

        # extrinsic - need to provide external corpus
        # cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_uci')
        # uci = cm.get_coherence()

        result_ = [vocab_size, test_vocab, max_features, n_topics, test_perplexity, u_mass]
        topic_coherence.append(result_)
        result = pd.DataFrame(topic_coherence, columns=cols).sort_values('u_mass')
    elapsed = time() - start
    print(f'\nDone: {i/n:.2%} | Duration: {format_time(elapsed)} | To Go: {format_time(elapsed/i*(n-i))}\n')
    result = pd.DataFrame(topic_coherence, columns=cols).sort_values('u_mass')
    print(result.head(10))
    result.to_csv(experiment_path / 'coherence.csv', index=False)
