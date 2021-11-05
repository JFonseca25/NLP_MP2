import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils import LBL_COL, Q_COL, A_COL
from l_model import LModel

class LR(LModel):
    def __init__(self, training_set: pd.DataFrame, vectorizer=None) -> None:
        super().__init__(training_set, vectorizer=vectorizer, model=LogisticRegression(max_iter=1000))
    
    def train(self):
        labels = self.training_set[LBL_COL]

        corpus = []
        for ix, row in self.training_set.iterrows():
            to_add = ' '.join([row[Q_COL], row[A_COL]])
            corpus.append(to_add)

        X = self.vectorizer.fit_transform(corpus)
        X = StandardScaler(with_mean=False).fit_transform(X)

        self.model.fit(X, labels)

    def predict(self, test_set: pd.DataFrame):
        tests = []

        for ix, row in test_set.iterrows():
            to_add = ' '.join([row[Q_COL], row[A_COL]])
            tests.append(to_add)
        
        X = self.vectorizer.transform(tests)
        X = StandardScaler(with_mean=False).fit_transform(X)

        return self.model.predict(X)