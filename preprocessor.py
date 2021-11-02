from _typeshed import Self
import pandas as pd
from .utils import *

class Preprocessor():
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

    def process_corpus_to_dict(self, corpus_path=None, data_frame=None):
        """
        Pre-processes a corpus and returns a dictionary whose keys are the
        CATEGORIES (see utils.py) and whose values are sets of pre-processed
        tokens.
        """
        
        # --- Auxiliary Functions (pretty code, yay) ---

        def get_question_and_answer(row):
            return row[Q_COL], row[A_COL]

        def add_tokens_to_dict(d, row, tokens):
            label = row[LBL_COL]
            d[label] = d[label].union(tokens)

        # --- Actual function ---

        assert corpus_path is not None or data_frame is not None

        if (data_frame is None):
            corpus = pd.read_csv(corpus_path, sep = "\t", names = TRAINING_COLS)
        else:
            corpus = data_frame
        dic = dict((key, set()) for key in CATEGORIES)

        for index, row in corpus.iterrows():
            print(index)

            question, answer = get_question_and_answer()

            tokens = word_tokenize(question) + word_tokenize(answer)

            processed_tokens = self.process_tokens(tokens)

            add_tokens_to_dict(dic, row, processed_tokens)
    
        return dic
