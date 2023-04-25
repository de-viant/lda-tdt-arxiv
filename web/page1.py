import pandas as pd

df = pd.read_csv('../data/meta_data_clean.csv')
print('read data complete..')
total_rows = df.shape[0]


# grouped_dataframe_author = df.groupby(['author'], sort=True, )
# API 1
def get_overall_papers_count():
    return total_rows


# API 2
def get_papers_category_count():
    return df['category'].value_counts().head(5).to_dict()


# API 3
def get_publication_frequency_timeline():
    df['published'] = pd.to_datetime(df.published, format='%Y-%m-%d %H:%M:%S')
    ans = df.groupby(df['published'].dt.year).agg(['count'])['id']
    return ans.sort_values(by=['count'], ascending=False).head().to_dict()['count']


# API 4
def get_tokens_count():
    return 1500


# API 5
def get_top_authors():
    return df['author'].value_counts().head().index.tolist()


def get_publication_frequency_timeline_by_category(cat):
    df_filtered = df[df['category'] == cat]
    df_filtered['published'] = pd.to_datetime(df_filtered.published, format='%Y-%m-%d %H:%M:%S')
    ans = df_filtered.groupby(df_filtered['published'].dt.year).agg(['count'])['id']
    return ans.sort_values(by=['count'], ascending=False).head().to_dict()['count']


def get_top_authors_and_publications():
     return df['author'].value_counts().head().to_dict()


# API 6
def get_top_authors_and_publications_by_category(cat):
    df_filtered = df[df['category'] == cat]
    return df_filtered['author'].value_counts().head().to_dict()


def get_top_words():
    return ['image', 'network', 'feature', 'dataset', 'training', 'learn', 'sample', 'layer', 'matrix', 'point ']
    # return ['image', 'network', 'feature', 'dataset', 'training']


def get_bottom_words():
    return ['albeit', 'remarkably', 'involved', 'ghz', 'noticeable', 'devote', 'faculty', 'seminal', 'fortunately',
            'terms']


def get_papers_category_count_by_category(cat):
    df_filtered = df[df['category'] == cat]
    return df_filtered['category'].value_counts().head(5).to_dict()
