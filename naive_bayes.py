import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from l_model import LModel
from utils import *

class NaiveBayes(LModel):
    def __init__(self, training_set: pd.DataFrame, vectorizer = TfidfVectorizer(stop_words='english')):
        super().__init__(training_set, vectorizer)
        self.model = MultinomialNB()

    def train(self):
        labels = self.training_set[LBL_COL]

        corpus = []
        for ix, row in self.training_set.iterrows():
            to_add = ' '.join([row[Q_COL], row[A_COL]])
            corpus.append(to_add)

        X = self.vectorizer.fit_transform(corpus)
        
        self.model.fit(X, labels)

    def test(self, test_set: pd.DataFrame):
        tests = []
        true_labels = []

        for ix, row in test_set.iterrows():
            to_add = ' '.join([row[Q_COL], row[A_COL]])
            tests.append(to_add)
            true_labels.append(row[LBL_COL])
        
        X = self.vectorizer.fit_transform(tests)

        return true_labels, self.model.predict(X)
