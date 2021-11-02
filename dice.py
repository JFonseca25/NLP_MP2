import numpy as np

from sklearn.metrics import jaccard_score
from .bow_model import BOWModel
from .utils import *

class Dice(BOWModel):
    def __init__(self, training_set: dict) -> None:
        super().__init__(training_set = training_set)

    def dice_score(y_true: np.array, y_pred: np.array):
        jaccard = jaccard_score(y_true, y_pred)
        return 2 * jaccard / (1 + jaccard)

    def test_tokens(self, string: str) -> str:
        max_dice = 0
        best_cat = ""

        tokens = word_tokenize(string)

        for cat in CATEGORIES:
            words_cat: set = self.training_set[cat]
            cat_dice = self.dice_score(list(words_cat), tokens)

            if cat_dice > max_dice:
                max_dice = cat_dice
                best_cat = cat

        return best_cat
    