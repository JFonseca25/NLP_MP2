import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *

class LModel:
    def __init__(self, training_set: pd.DataFrame, vectorizer = None, model=None) -> None:
        self.training_set: pd.DataFrame = training_set

        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(stop_words='english')
        else:
            self.vectorizer = vectorizer

        self.model = model
    
    def train(self):
        labels = self.training_set[LBL_COL]

        corpus = []
        for ix, row in self.training_set.iterrows():
            to_add = ' '.join([row[Q_COL], row[A_COL]])
            corpus.append(to_add)

        X = self.vectorizer.fit_transform(corpus)

        self.model.fit(X, labels)

    def predict(self, test_set: pd.DataFrame):
        tests = []

        for ix, row in test_set.iterrows():
            to_add = ' '.join([row[Q_COL], row[A_COL]])
            tests.append(to_add)
        
        X = self.vectorizer.transform(tests)

        return self.model.predict(X)
