import numpy as np

from bow_model import BOWModel
from utils import *

class Dice(BOWModel):
    def __init__(self, training_set: dict) -> None:
        super().__init__(training_set = training_set)

    def dice_distance(self, set1: set, set2: set):
        return (2 * len(set1.intersection(set2))) / (len(set1) + len(set2))

    def test_tokens(self, string: str) -> str:
        max_dice = 0
        best_cat = ""

        tokens = word_tokenize(string)
        tokens = process_tokens(tokens)

        for cat in CATEGORIES:
            words_cat = self.bow[cat]
            cat_dice = self.dice_distance(words_cat, set(tokens))

            if cat_dice > max_dice:
                max_dice = cat_dice
                best_cat = cat

        return best_cat
    