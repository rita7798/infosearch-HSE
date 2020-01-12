import pickle
import os
from bm25 import bm25_search
from fasttext import fasttext_search
from tf_idf import tf_idf_search
from flask import Flask, render_template, request


app = Flask(__name__)


@app.route('/')
def main_page():
    return render_template('index.html')


@app.route('/results')
def results():
    query = request.args['q']
    if request.args['model'] == 'tf-idf':
        res = tf_idf_search(query)[:10]
    elif request.args['model'] == 'BM25':
        res = bm25_search(query)[:10]
    elif request.args['model'] == 'fasttext':
        res = fasttext_search(query)[:10]
    response = []
    for i in res:
        response.append(str(i[0] +  ', score = ' + str(i[1])))
    return render_template('results.html', query=query, response=response) 
                     
    
if __name__ == '__main__':
    app.run(debug=True)
