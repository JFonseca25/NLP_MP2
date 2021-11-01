import pandas as pd
import numpy as np
from .utils import *

class Model():
    def __init__(self, training_set: dict = None) -> None:
        
        self.training_set: dict = training_set
        """
        A dictionary whose keys are the categories and whose values are whatever is needed
        """

    def set_training_set(self, training_set: dict) -> None:
        self.training_set = training_set
    
    def pre_process_row(self, question: str, answer: str) -> list[str]:
        """
        Pre-processes a row and returns a list of its tokens.
        """
        tokens = word_tokenize(question) + word_tokenize(answer)
        return tokens

    def test_tokens(self, tokens: list[str]) -> str:
        """
        Tests the given tokens and returns a category from CATEGORIES.
        """
        raise NotImplementedError("Token test strategy not implemented for class " + self.__class__.__name__)

    def test_row(self, row):
        tokens = self.pre_process_row(row[Q_ROW], row[A_ROW])
        return self.test_tokens(tokens)
    
    def test(self, test_set: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Runs the model on the given test set and returns (true_answers, pred_answers).
        """

        true_answers = []
        pred_answers = []
        
        for ix, row in test_set.iterrows():
            true_answers.append(row[LBL_ROW])
            pred_answers.append(self.test_row(row))

        return true_answers, pred_answers
        