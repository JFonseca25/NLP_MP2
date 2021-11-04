import pandas as pd

from sklearn import svm
from l_model import LModel

class SVM(LModel):
    def __init__(self, training_set: pd.DataFrame, vectorizer=None, model=None):
        super().__init__(training_set=training_set, vectorizer=vectorizer, model=model)

        if model is None:
            self.model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    