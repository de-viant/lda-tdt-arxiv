from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
from pathlib import Path
import topic_modeling_vis as tmv
import similar_papers as sp
import json
import page1

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

## prefetch loading of csv files for doc ##

def getTableBodyFromCSV(topics,min_df,max_features):
    topic_dist_key = f'{int(min_df*1000):d}_{max_features}_{topics}'
    topic_dist_path = Path('../data/topic_distribution') / f'{topic_dist_key}.csv'
    print("file path:"+str(topic_dist_path))
    df = pd.read_csv(topic_dist_path, header=None, na_values=[''], keep_default_na=False)
    df = df.transpose()
    df = df.sample(100)
    tableBody = ""
    for index, row in df.iterrows():
        if (index == 0):
            continue
        tableBody = tableBody + "<tr><td>" + row[0][:-4] + "</td>"
        for i in range(topics):
            row_value = str(row[i + 1])
            if (row_value == "nan"):
                row_value = "0"
            else:
                row_value = row_value[:7]
            tableBody = tableBody + "<td>" + row_value + "</td>"
        tableBody = tableBody + "</tr>"
    return tableBody


topic_list=[5, 10, 20]
min_df_list =[0.001]
max_features_list = [10000, 25000]
cache ={}



for t in topic_list:
    for m in min_df_list:
        for f in max_features_list:
            topic_dist_key = f'{int(m*1000):d}_{f}_{t}'
            print("calculating value for : "+topic_dist_key)
            table_body = getTableBodyFromCSV(t,m,f)
            print("value calculated for : "+topic_dist_key)
            cache[topic_dist_key]=table_body


def getTableBodyFromCache(topics,min_df,max_features):
    topic_dist_key = f'{int(min_df*1000):d}_{max_features}_{topics}'
    return cache[topic_dist_key]


## prefetch end


@app.route("/")
@app.route("/summary")
def summary():
    cat = request.args.get('category')
    data_values = {'overall_papers_count': page1.get_overall_papers_count()}
    data_values["top_words"] = page1.get_top_words()
    data_values["bottom_words"] = page1.get_bottom_words()

    if cat is None or cat == '':
        category_values = page1.get_papers_category_count()
        top_authors = page1.get_top_authors_and_publications()
        pub_freq = page1.get_publication_frequency_timeline()
    else:
        category_values = page1.get_papers_category_count_by_category(cat)
        top_authors = page1.get_top_authors_and_publications_by_category(cat)
        pub_freq = page1.get_publication_frequency_timeline_by_category(cat)

    data_values["top_authors"] = top_authors
    data_values["category_values"] = category_values

    data_values["publication_values"] = pub_freq
    data_values["doc_len_histogram_url"] = "document_1.png"
    data_values["unique_tokens_histogram_url"] = "document_2.png"
    data_values["wordcloud_url"] = "wordCloud.png"

    data_string = json.dumps(data_values, sort_keys=True, indent=4, separators=(',', ': '))

    # data_values["html1"] = createHtml(data_values);

    return render_template('app.html', **locals())


@app.route("/topic-models")
def topicmodels():
    topics = 5
    min_df = 0.001
    max_features = 10000
    ldavis = tmv.getLDAvis(topics, float(min_df), max_features)
    return render_template('app.html', topics=topics, min_df=min_df, max_features=max_features, ldavis=ldavis)


@app.route("/topic-models/view/topics/<topics>/min_df/<min_df>/max_features/<max_features>")
def topicmodelsview(topics, min_df, max_features):
    ldavis = tmv.getLDAvis(topics, float(min_df), max_features)
    return render_template('app.html', topics=topics, min_df=min_df, max_features=max_features, ldavis=ldavis)


@app.route("/docs/view/topics/<topics>/min_df/<min_df>/max_features/<max_features>/title/<title>")
def doc_details(topics, min_df, max_features, title):
    # Paper details
    meta_data_path = Path('../data/meta_data.csv')
    meta_data = pd.read_csv(meta_data_path, index_col=0)
    meta_data = meta_data.loc[meta_data['title'] == title]
    # Similar papers
    similar_papers = sp.get_similar_papers(topics, float(min_df), max_features, title)
    if meta_data.size == 0:
        return render_template('doc-details.html', err = "No Data Found", similar=similar_papers)
    else:
        return render_template('doc-details.html', title = title, summary = meta_data.ix[0,0], url = meta_data.ix[0,4], author=meta_data.ix[0,5], published=meta_data.ix[0,7] , similar=similar_papers)

@app.route("/docs")
def docs():
    # Default values
    topics = 5
    min_df = 0.001
    max_features = 10000
    tableBody = getTableBodyFromCache(topics, min_df, max_features)
    tableHeader = getTableHeader(topics)
    return render_template('app.html', tableBody=tableBody, tableHeader=tableHeader, topics=topics, min_df=min_df,
                           max_features=max_features)


@app.route("/docs/topics/<topics_v>/min_df/<min_df_v>/max_features/<max_features_v>")
def docs_specific(topics_v, min_df_v, max_features_v):
    # Default values
    topics = int(topics_v)
    min_df = float(min_df_v)
    max_features = int(max_features_v)

    tableBody = getTableBodyFromCache(topics,min_df,max_features)

    tableHeader = getTableHeader(topics)

    return render_template('app.html', tableBody=tableBody, tableHeader=tableHeader, topics=topics, min_df=min_df,
                           max_features=max_features)


def getTableHeader(topics):
    tableHeader = "<th>Name</th>"
    for i in range(topics):
        tableHeader = tableHeader + "<th>Topic " + str(i + 1) + "</th>"
    return tableHeader


if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
