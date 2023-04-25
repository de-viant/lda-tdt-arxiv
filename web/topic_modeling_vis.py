from pathlib import Path
import pandas as pd

from scipy import sparse
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.test.utils import datapath

import pyLDAvis
from pyLDAvis.gensim import prepare

experiment_path = Path('../topic_modeling/experiments_06')
corpus_path = experiment_path / 'corpora'

def getLDAvis(topics, min_df, max_features):
    ldavis_key = f'{int(min_df*1000):d}_{max_features}_{topics}'
    ldavis_path = Path('./pyldavis') / f'{ldavis_key}_tsne.html'
    if not ldavis_path.exists():
        key = f'{max_features}'
        dtm_path = corpus_path / f'dtm_{key}.npz'
        dtm = sparse.load_npz(dtm_path)
        token_path = corpus_path / f'tokens_{key}.csv'
        tokens = pd.read_csv(token_path, header=None, squeeze=True, na_values=[], keep_default_na=False)
        model_file = datapath((experiment_path / 'models' / f'{key}_{topics}').resolve())
        lda_model = LdaModel.load(model_file)
        id2word = tokens.to_dict()
        corpus = Sparse2Corpus(dtm, documents_columns=False)
        dictionary = Dictionary.from_corpus(corpus, id2word)
        vis = prepare(lda_model, corpus, dictionary, mds='tsne')
        kwargs = {"ldavis_url": "/static/ldavis.js"}
        pyLDAvis.save_html(vis, str(ldavis_path), **kwargs)
    with open(str(ldavis_path), 'r') as myfile:
        data = myfile.read()
    return data

#getLDAvis(5, 0.001, 10000)
#getLDAvis(5, 0.001, 25000)
#getLDAvis(10, 0.001, 10000)
#getLDAvis(10, 0.001, 25000)
#getLDAvis(20, 0.001, 10000)
#getLDAvis(20, 0.001, 25000)