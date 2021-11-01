import pandas as pd

from sklearn.naive_bayes import MultinomialNB

from embeddings_model import EModel
from utils import *

class NaiveBayes(EModel):
    def __init__(self, training_set: pd.DataFrame = None, wd_mat=None) -> None:
        super().__init__(training_set=training_set, wd_mat=wd_mat)
        self.model = MultinomialNB()

    def test_row(self, row):
        doc = ' '.join([row[Q_ROW], row[A_ROW]])
        vectorized_doc = self.vectorize_doc(doc)

    def train(self):
        wd_mat = self.get_wd_mat()
