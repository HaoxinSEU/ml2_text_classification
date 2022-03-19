import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')

from nltk import word_tokenize
from nltk.stem import PorterStemmer

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

import csv
import time

"""do preprocessing, delete useless symbols, then tokenize"""
def DeleteSymbolAndTokenize(text):
    # delete meaningless symbols
    text = re.sub("(<.*?>)", "", text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"(#[\d\w\.]+)", '', text)
    text = re.sub(r"(@[\d\w\.]+)", '', text)
    text = re.sub("(\\W|\\d)", " ", text)

    # delete header and tailer
    text = text.strip()

    # tokenize words
    text = word_tokenize(text)

    # stem
    porter = PorterStemmer()
    stem_text = [porter.stem(word) for word in text]

    return stem_text


def run():
    # load data, then divide into training data and validation data
    data_train_neg = pd.read_csv('../datasets/train_neg_full.txt', sep='\r', names=['text'], header=None, encoding='utf-8')
    data_train_neg.insert(data_train_neg.shape[1], "labels", -1)

    data_train_pos = pd.read_csv('../datasets/train_pos_full.txt', sep='\r', names=['text'], header=None, encoding='utf-8')
    data_train_pos.insert(data_train_pos.shape[1], "labels", 1)

    data_train = pd.concat([data_train_neg, data_train_pos], ignore_index=True)
    data_train = data_train.sample(frac=1).reset_index(drop=True)

    data_eval = data_train.iloc[2400000:-1, :]
    data_train = data_train.iloc[:2400000, :]

    X_train = data_train['text'].values
    y_train = data_train['labels'].values

    X_eval = data_eval['text'].values
    y_eval = data_eval['labels'].values

    all_data = pd.concat([data_train, data_eval])


    # load test data
    data_test = pd.read_csv('../datasets/test_data.txt', sep='\r', names=['text'], header=None, encoding='utf-8')
    X_test = data_test['text'].values
    starttime = time.time()

    # using TFIDF
    vect = TfidfVectorizer(tokenizer=DeleteSymbolAndTokenize, sublinear_tf=True, norm='l2', ngram_range=(1, 2))
    vect.fit_transform(all_data.text)
    # transform text to vector
    X_train_vect = vect.transform(X_train)
    
    # train the Linear SVC model
    svc = LinearSVC(tol=1e-05)
    svc.fit(X_train_vect, y_train)

    train_endtime = time.time()
    print("training time: ", train_endtime - starttime)

    # evaluate the model
    X_eval_vect = vect.transform(X_eval)
    y_eval_pred = svc.predict(X_eval_vect)
    eval_endtime = time.time()
    print("evaluation time: ", eval_endtime - train_endtime)

    print("Prediction accuracy on evaulation set: ", accuracy_score(y_eval, y_eval_pred))
    print("F1 Score on evaulation set: ", f1_score(y_eval, y_eval_pred, average='micro'))

    # use the model to do prediction
    X_test_vect = vect.transform(X_test)
    predict_res = svc.predict(X_test_vect)

    # write to the .csv file
    test_id = np.arange(1, 10001)

    print("writing results to csv...")
    with open('../prediction.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(test_id, predict_res):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
    print("finished!")

if __name__ == "__main__":
    run()
