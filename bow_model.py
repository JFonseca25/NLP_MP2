import pandas as pd
from utils import A_COL, Q_COL

class BOWModel:
    def __init__(self, training_set: dict) -> None:
        assert training_set is not None
        self.bow = training_set

    def test_tokens(self, string: str):
        raise NotImplementedError

    def test(self, test_set: pd.DataFrame):
        pred = []

        for ix, row in test_set.iterrows():
            pred.append(self.test_tokens(' '.join([row[Q_COL], row[A_COL]])))

        return pred
