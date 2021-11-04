import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from l_model import LModel
from utils import *

class NaiveBayes(LModel):
    def __init__(self, training_set: pd.DataFrame, vectorizer=None, model=None):
        super().__init__(training_set=training_set, vectorizer=vectorizer, model=model)

        if model is None:
            self.model = MultinomialNB()
