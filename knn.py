import pandas as pd

from l_model import LModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class KNN(LModel):
    def __init__(self, training_set: pd.DataFrame, vectorizer=None, model=None, n_neighs = 21):
        super().__init__(training_set=training_set, vectorizer=vectorizer, model=model)

        if model is None:
            self.model = KNeighborsClassifier(weights='distance')

        self.model.set_params(n_neighbors=n_neighs)
