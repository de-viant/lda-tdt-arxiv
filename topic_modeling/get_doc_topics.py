from pathlib import Path
import pandas as pd

from scipy import sparse
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.test.utils import datapath


experiment_path = Path('../topic_modeling/experiments_06')
corpus_path = experiment_path / 'corpora'

def createTopicDistribution(topics, min_df, max_features):
    topic_dist_key = f'{int(min_df*1000):d}_{max_features}_{topics}'
    topic_dist_path = Path('../data/topic_distribution') / f'{topic_dist_key}.csv'
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

    text_path = Path('../data/clean_stop')
    text_files = text_path.glob('*.txt')
    docs = [(f.name, f.read_text()) for f in text_files]

    topic_labels = [f'Topic {i}' for i in range(1, topics + 1)]
    document_topics = pd.DataFrame(index=topic_labels)

    for i, doc in enumerate(docs):
        bow = dictionary.doc2bow(doc[1].split())
        document_topics[doc[0]] = pd.Series({f'Topic {k+1}': v for k, v in lda_model.get_document_topics(bow=bow, minimum_probability=1e-3)})

    document_topics.to_csv(topic_dist_path)


#createTopicDistribution(5, 0.001, 10000)
#createTopicDistribution(5, 0.001, 25000)
#createTopicDistribution(10, 0.001, 10000)
#createTopicDistribution(10, 0.001, 25000)
#createTopicDistribution(20, 0.001, 10000)
#createTopicDistribution(20, 0.001, 25000)

