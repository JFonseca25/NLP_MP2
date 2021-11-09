import sys
import argparse
import pandas as pd

from utils import *
from sklearn import svm
from lemma_tokenizer import LemmaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

def run_model(training_set: pd.DataFrame, test_set: pd.DataFrame):
    def get_corpus(training_set):
        corpus = []
        for ix, row in training_set.iterrows():
            string = ' '.join([row[Q_COL], row[A_COL]])
            corpus.append(string)
        return corpus

    corpus_train = get_corpus(training_set)
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer())
    X_train = vectorizer.fit_transform(corpus_train)
    y_train = training_set[LBL_COL]

    corpus_test = get_corpus(test_set)
    X_test = vectorizer.transform(corpus_test)

    model = svm.LinearSVC()
    model.fit(X_train, y_train)

    return model.predict(X_test)

def main(training_set: pd.DataFrame, test_set: pd.DataFrame):
    pred_labels = run_model(training_set, test_set)
    for label in pred_labels:
        print(label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a model, classifies the test set and outputs a classification.")
    parser.add_argument("-test", help="The file to test the model on.")
    parser.add_argument("-train", help="The file to train the model on.")
    args = parser.parse_args(sys.argv[1:])

    training_set = pd.read_table(args.train, sep='\t', names=[LBL_COL, Q_COL, A_COL], 
                                    converters={Q_COL: str, A_COL: str})
    
    test_set = pd.read_table(args.test, sep='\t', names=[Q_COL, A_COL],
                                    converters={Q_COL: str, A_COL: str})

    main(training_set, test_set)
