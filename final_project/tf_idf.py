import csv
import numpy as np
import os
import pickle
import pymorphy2
import re
from math import log
from nltk.corpus import stopwords
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer


m = pymorphy2.MorphAnalyzer()
stopwords = stopwords.words("russian")
vectorizer = CountVectorizer()


def get_docs():
    with open('quora_question_pairs_rus.csv', 'r', encoding='utf-8') as t:
        corpus = csv.reader(t)
        text = list(corpus)
    if not os.path.exists('docs.pickle'):
        docs = []
        for i, line in enumerate(text):
            if i != 0 and i < 10002:
                docs.append(line[2])
        with open("docs.pickle", "wb") as p:
                pickle.dump(docs, p)
    else:
        with open("docs.pickle", "rb") as p:
            docs = pickle.load(p)
    return docs


def get_corpus(docs):        
    if not os.path.exists('corpus.pickle'):
        corpus = []
        for i, sent in enumerate(docs):
            if i < 10001:
                corpus.append(' '.join(clean_text(sent)))
            else:
                break
            with open("corpus.pickle", "wb") as d:
                pickle.dump(corpus, d)
    else:
        with open("corpus.pickle", "rb") as d:
            corpus = pickle.load(d)
    return corpus


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'[A-Za-z]', '', text)
    text = [m.normal_forms(x)[0] for x in text.split() if x not in stopwords]
    return text


def tf_idf_matrix(tf_matrix, N, all_ns, doc_words):
    tf_idf_matrix = []
    for i, word in enumerate(doc_words):
        tf_idf_matrix.append([])
        for doc in corpus:
            tf_idf = 0
            if word in doc:
                w_i = doc_words.index(word)
                d_i = corpus.index(doc)
                tf = tf_matrix[w_i][d_i]/len(doc)
                n = all_ns[word]
                idf = log(N/n)
                tf_idf = tf*idf
            tf_idf_matrix[i].append(tf_idf)
    return np.array(tf_idf_matrix)


def upload_tf_idf_matrix(tf_matrix, N, all_ns, doc_words):
    if os.path.exists('tf_idf.pickle'):
        with open("tf_idf.pickle", "rb") as d:
            tf_idf_matrix = pickle.load(d)
    else:
        tf_idf_matrix = tf_idf_matrix(tf_matrix, N, all_ns, doc_words)
        with open("tf_idf.pickle", "wb") as d:
            pickle.dump(tf_idf_matrix, d)
    return tf_idf_matrix


def query2vec_tf_idf(query, doc_words, N, all_ns):
    query = clean_text(query)
    counts = {}
    for word in query:
        if word not in counts:
            counts[word] = 1
        else:
            counts[word] += 1
    v_query = []
    for word in doc_words:
        if word in counts:
            tf = counts[word]/len(query)
            n = all_ns[word]
            idf = log((N+1)/(n+1))
            tf_idf = tf*idf
            v_query.append(tf_idf)
        else:
            v_query.append(0)
    return np.array(v_query)


def tf_idf_search(query):
    docs = get_docs()
    corpus = get_corpus(docs)        
    N = len(corpus)
    X = vectorizer.fit_transform(corpus)
    f_matrix = X.toarray()
    tf_matrix = np.transpose(f_matrix)
    doc_words = vectorizer.get_feature_names()
    all_ns = {}
    for word in doc_words:
        w_i = doc_words.index(word)
        all_ns[word] = np.count_nonzero(tf_matrix[w_i])
    v_query = query2vec_tf_idf(query, doc_words, N, all_ns)
    tf_idf_matrix = upload_tf_idf_matrix(tf_matrix, N, all_ns, doc_words)
    doc_score = v_query.dot(tf_idf_matrix)
    response = list(zip(docs, doc_score))
    response = sorted(response, key=itemgetter(1), reverse=True)
    return response
