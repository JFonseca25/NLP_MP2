from sklearn.metrics import jaccard_score

from preprocessor import Preprocessor
from .model import Model
from .utils import *

class Jaccard(Model):
    def __init__(self, training_set: dict = None) -> None:
        super().__init__(training_set = training_set)

    def pre_process_row(self, question: str, answer: str) -> list[str]:
        tokens = super().pre_process_row(question, answer)

        preprocessor = Preprocessor()
        tokens = preprocessor.lowercasing().sw_removal().lemmatize().process_tokens(tokens)

        return tokens

    def test_tokens(self, tokens: list[str]) -> str:
        max_jaccard = 0
        best_cat = ""

        for cat in CATEGORIES:
            words_cat: set = self.training_set[cat]
            cat_jacc = jaccard_score(list(words_cat), tokens)

            if cat_jacc > max_jaccard:
                max_jaccard = cat_jacc
                best_cat = cat

        return best_cat
    