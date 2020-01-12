import csv
import numpy as np
import pickle
import pymorphy2
import re
import os
from gensim.models.keyedvectors import KeyedVectors
from numpy import dot
from numpy.linalg import norm
from operator import itemgetter


ft_model = KeyedVectors.load('models/model.model')
m = pymorphy2.MorphAnalyzer()


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
    if not os.path.exists('vectors.pickle'):
        corpus = []
        for i, sent in enumerate(docs):
            if i < 10001:
                corpus.append(clean_text(sent))
            else:
                break
            with open("vectors.pickle", "wb") as p:
                pickle.dump(corpus, p)
    else:
        with open("vectors.pickle", "rb") as p:
            corpus = pickle.load(p)


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = [m.normal_forms(x)[0] for x in text.split()]
    return text


def ft_matrix(corpus):
    matrix = []
    for doc in corpus:
        doc_vectors = np.zeros((len(doc), ft_model.vector_size))
        vec = np.zeros((ft_model.vector_size,))
        for i, lemma in enumerate(doc):
            if lemma in ft_model.vocab:
                doc_vectors[i] = ft_model.wv[lemma]
        if doc_vectors.shape[0] is not 0:
            vec = np.mean(doc_vectors, axis=0)
        matrix.append(vec)
    return np.array(matrix)


def upload_ft_matrix(corpus):
    if os.path.exists('fasttext.pickle'):
        with open("fasttext.pickle", "rb") as p:
            ft_matrix = pickle.load(p)
    else:
        ft_matrix = ft_matrix(corpus)
        with open("fasttext.pickle", "wb") as p:
            pickle.dump(ft_matrix, p)
    return ft_matrix


def query2vec(query):
    query = clean_text(query)
    lemmas_vectors = np.zeros((len(query), ft_model.vector_size))
    vec = np.zeros((ft_model.vector_size,))
    for i, lemma in enumerate(query):
        if lemma in ft_model.vocab:
            lemmas_vectors[i] = ft_model.wv[lemma]
    if lemmas_vectors.shape[0] is not 0:
        vec = np.array(np.mean(lemmas_vectors, axis=0))
    return vec


def fasttext_search(query):
    response = []
    vec = query2vec(query)
    docs = get_docs()
    corpus = get_corpus(docs)
    ft_matrix = upload_ft_matrix(corpus)
    for i, doc in enumerate(docs):
        if i < len(ft_matrix):
            doc_score = dot(vec, ft_matrix[i])/(norm(vec)*norm(ft_matrix[i]))
            response.append((docs[i], doc_score))
    response = sorted(response, key=itemgetter(1), reverse=True)
    return response
