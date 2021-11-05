import pandas as pd

from sklearn.linear_model import Perceptron
from l_model import LModel

class PerceptronClassifier(LModel):
    def __init__(self, training_set: pd.DataFrame, vectorizer=None) -> None:
        super().__init__(training_set, vectorizer=vectorizer, model=Perceptron())