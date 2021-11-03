import numpy as np

from bow_model import BOWModel
from utils import *

class Jaccard(BOWModel):
    def __init__(self, training_set: dict) -> None:
        super().__init__(training_set = training_set)

    def jaccard_distance(self, set1: set, set2: set):
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def test_tokens(self, string: str) -> str:
        max_jacc = 0
        best_cat = ""

        tokens = word_tokenize(string)
        tokens = process_tokens(tokens)

        for cat in CATEGORIES:
            words_cat = self.bow[cat]
            cat_jacc = self.jaccard_distance(words_cat, set(tokens))

            if cat_jacc > max_jacc:
                max_jacc = cat_jacc
                best_cat = cat

        return best_cat
    