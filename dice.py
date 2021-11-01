import numpy as np

from preprocessor import Preprocessor
from sklearn.metrics import jaccard_score
from .model import Model
from .utils import *

class Dice(Model):
    def __init__(self, training_set: dict = None) -> None:
        super().__init__(training_set = training_set)

    def pre_process_row(self, question: str, answer: str) -> list[str]:
        tokens = super().pre_process_row(question, answer)

        preprocessor = Preprocessor()
        tokens = preprocessor.lowercasing().sw_removal().lemmatize().process_tokens(tokens)

        return tokens

    def dice_score(y_true: np.array, y_pred: np.array):
        jaccard = jaccard_score(y_true, y_pred)
        return 2 * jaccard / (1 + jaccard)

    def test_tokens(self, tokens: list[str]) -> str:
        max_dice = 0
        best_cat = ""

        for cat in CATEGORIES:
            words_cat: set = self.training_set[cat]
            cat_jacc = self.dice_score(list(words_cat), tokens)

            if cat_jacc > max_dice:
                max_dice = cat_jacc
                best_cat = cat

        return best_cat
    