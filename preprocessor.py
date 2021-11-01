from _typeshed import Self
import pandas as pd
from .utils import *

class Preprocessor():
    def __init__(self):
        self.lemmatize: bool = False
        self.sw_removal: bool = False
        self.lowercasing: bool = False

        return self

    def process_tokens(self, tokens: list[str]):
        """
        Returns a list of processed tokens, according to the specified options,
        without ponctuation.
        """

        # Remove ponctuation first
        ret = [word for word in tokens if word.isalpha()]

        if self.lowercasing:
            ret = [word.lower() for word in ret]
        elif self.sw_removal:
            ret = [word for word in ret if word not in stopwords.words()]
        elif self.lemmatize:
            wnl = WordNetLemmatizer()
            ret = [wnl.lemmatize(word) for word in ret]
        
        return ret

    def process_corpus(self, corpus_path):
        """
        Pre-processes a corpus and returns a dictionary whose keys are the
        CATEGORIES (see utils.py) and whose values are sets of pre-processed
        tokens.
        """
        
        # --- Auxiliary Functions (pretty code, yay) ---

        def get_question_and_answer(row):
            return row[Q_ROW], row[A_ROW]

        def add_tokens_to_dict(d, row, tokens):
            label = row[LBL_ROW]
            d[label] = d[label].union(tokens)

        # --- Actual function ---

        corpus = pd.read_csv(corpus_path, sep = "\t", names = TRAINING_ROWS)
        dic = dict((key, set()) for key in CATEGORIES)

        for index, row in corpus.iterrows():
            print(index)

            question, answer = get_question_and_answer()

            tokens = word_tokenize(question) + word_tokenize(answer)

            processed_tokens = self.process_tokens(tokens)

            add_tokens_to_dict(dic, row, processed_tokens)
    
        return dic

    def sw_removal(self):
        self.sw_removal = True
        return self

    def lowercasing(self):
        self.lowercasing = True
        return self

    def lemmatize(self):
        self.lemmatize = True
        return self
