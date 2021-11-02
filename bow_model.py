import pandas as pd
from preprocessor import Preprocessor
from utils import A_COL, LBL_COL, Q_COL

class BOWModel:
    def __init__(self, training_set: dict) -> None:
        assert training_set is not None
        self.bow = training_set

    def test_tokens(self, string: str):
        raise NotImplementedError

    def test(self, test_set: pd.DataFrame):
        true = []
        pred = []

        for ix, row in test_set.iterrows():
            true.append(row[LBL_COL])
            pred.append(self.test_tokens(' '.join([row[Q_COL], row[A_COL]])))

        return true, pred