import pandas as pd

class LModel:
    def __init__(self, training_set: pd.DataFrame, vectorizer = None) -> None:
        self.training_set: pd.DataFrame = training_set
        self.vectorizer = vectorizer
    
    def train(self):
        raise NotImplementedError

    def predict(self, test_set: pd.DataFrame):
        raise NotImplementedError
