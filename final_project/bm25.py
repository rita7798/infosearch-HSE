import csv
import nltk
import numpy as np
import pickle
import pymorphy2
import re
import os
from math import log
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.text import Text
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


def bm25_matrix(all_ns, docs_len, tf_matrix, corpus, doc_words, avgdl, N):
    k = 2.0
    b = 0.75
    bm_matrix = []
    for i, word in enumerate(doc_words):
        bm_matrix.append([])
        for doc in corpus:
            bm = 0
            if word in doc:
                w_i = doc_words.index(word)
                d_i = corpus.index(doc)
                doc_len = docs_len[doc]
                tf = tf_matrix[w_i][d_i]/doc_len
                n = all_ns[word]
                idf = log((N-n+0.5)/(n+0.5))
                bm = idf*((tf*(k+1))/(tf+k*(1-b+(b*(doc_len/avgdl)))))
            bm_matrix[i].append(bm)
    return np.array(bm_matrix)


def upload_bm25_matrix(all_ns, docs_len, tf_matrix, corpus, doc_words, avgdl, N):
    if os.path.exists('bm25.pickle'):
        with open("bm25.pickle", "rb") as d:
            bm_matrix = pickle.load(d)
    else:
        bm_matrix = bm25_matrix(all_ns, docs_len, tf_matrix, corpus, doc_words, avgdl, N)
        with open("bm25.pickle", "wb") as d:
            pickle.dump(bm_matrix, d)
    return bm_matrix


def vect_bm25(query, k, doc_words, all_ns, N):
    q_words = clean_text(query)
    counts = {}
    for word in q_words:
        if word not in counts:
            counts[word] = 1
        else:
            counts[word] += 1
    v_query = []
    for word in doc_words:
        bm = 0
        if word in counts:
            doc_len = len(q_words)
            n = all_ns[word]
            tf = counts[word]/doc_len
            idf = log((N+1)/(n+1))
            bm = idf*((tf*(k+1))/(tf+k))
        v_query.append(bm)
    v_query = np.array(v_query)
    return v_query


def bm25_search(query):
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
    docs_len = {}
    whole_len = 0
    for doc in corpus:
        doc_len = len(doc.split())
        docs_len[doc] = doc_len
        whole_len += doc_len
    avgdl = whole_len/N
    doc_words = vectorizer.get_feature_names()
    bm_matrix = upload_bm25_matrix(all_ns, docs_len, tf_matrix, corpus, doc_words, avgdl, N)
    k = 2.0
    b = 0.75
    q_vect = vect_bm25(query, k, doc_words, all_ns, N)
    doc_score = q_vect.dot(bm_matrix)
    response = list(zip(docs, doc_score))
    response = sorted(response,key=itemgetter(1), reverse = True)
    return response
