import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from l_model import LModel

class RForest(LModel):
    def __init__(self, training_set: pd.DataFrame, vectorizer=None, model=None) -> None:
        super().__init__(training_set, vectorizer=vectorizer, model=RandomForestClassifier())