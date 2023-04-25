#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd
import pandas as pd

from pathlib import Path
import tempfile
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

nlp = spacy.load('en_core_web_sm')

# Combine spacy and linguistic utils stopwords
stop_words = pd.read_csv('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words',
                         header=None,
                         squeeze=True)

for stop_word in stop_words:
    STOP_WORDS.add(stop_word)

# where your text files are
text_path = Path('data/text')
text_files = text_path.glob('*.txt')


# where the clean version should go
clean_path = Path('data/clean')
if not clean_path.exists():
    clean_path.mkdir(exist_ok=True, parents=True)
for i, text_file in enumerate(text_files):
    if i % 100 == 0:
        print(i, end=' ', flush=True)

    doc = text_file.read_text()
    clean_doc = ' '.join([t.lemma_ for t in nlp(doc) if not any([t.is_stop,
                                                                 t.is_digit,
                                                                 not t.is_alpha,
                                                                 t.is_punct,
                                                                 t.is_space,
                                                                 t.lemma_ == '-PRON-'])])
    (clean_path / text_file.name).write_text(clean_doc)


def clean_with_sklear(files):
    """can also do this directly in countvectorizer but takes forever since can't run in parallel"""
    def tokenizer(doc):
        return [t.lemma_ for t in nlp(doc)
                if not any([t.is_stop,
                            t.is_digit,
                            not t.is_alpha,
                            t.is_punct,
                            t.is_space,
                            t.lemma_ == '-PRON-'])]

    vectorizer = CountVectorizer(tokenizer=tokenizer, binary=True)
    doc_term_matrix = vectorizer.fit_transform([f.read_text() for f in files])
