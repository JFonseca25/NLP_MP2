from sklearn.metrics import jaccard_score

from preprocessor import Preprocessor
from .bow_model import BOWModel
from .utils import *

class Jaccard(BOWModel):
    def __init__(self, training_set: dict) -> None:
        super().__init__(training_set = training_set)

    def test_tokens(self, string: str) -> str:
        max_jacc = 0
        best_cat = ""

        tokens = word_tokenize(string)

        for cat in CATEGORIES:
            words_cat: set = self.training_set[cat]
            cat_jacc = jaccard_score(list(words_cat), tokens)

            if cat_jacc > max_jacc:
                max_jacc = cat_jacc
                best_cat = cat

        return best_cat
    