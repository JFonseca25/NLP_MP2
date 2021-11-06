import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

"""Categories in the training corpus"""
CATEGORIES = ["GEOGRAPHY", "MUSIC", "LITERATURE", "HISTORY", "SCIENCE"]

"""Label for the Label row of the training set"""
LBL_COL = "l"

"""Label for the Question Row of the training and testing set"""
Q_COL = "q"

"""Label for the Answer Row of the training and testing sets"""
A_COL = "a"

STOPWORDS = set(stopwords.words('english'))

COLS = [LBL_COL, Q_COL, A_COL]


def process_tokens(tokens: list[str]):
    """
    Returns a list of processed tokens, lemmatized, lowercased, and without
    ponctuation.
    """
    
    wnl = WordNetLemmatizer()

    # lowercased words without punctuation
    ret = [word.lower() for word in tokens if word.isalpha()]
    # lemmatized words that are not stopwords
    ret = [wnl.lemmatize(word) for word in ret if word not in stopwords.words()]
    
    return ret
