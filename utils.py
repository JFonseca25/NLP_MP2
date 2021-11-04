import nltk
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

"""Categories in the training corpus"""
CATEGORIES = ["GEOGRAPHY", "MUSIC", "LITERATURE", "HISTORY", "SCIENCE"]

"""The relative path of the training file"""
TRAINING_FILE = "trainWithoutDev.txt"

"""The relative path of the testing file"""
TESTING_FILE = "dev_clean.txt"

"""Label for the Label row of the training set"""
LBL_COL = "l"

"""Label for the Question Row of the training and testing set"""
Q_COL = "q"

"""Label for the Answer Row of the training and testing sets"""
A_COL = "a"

STOPWORDS = set(stopwords.words('english'))

COLS = [LBL_COL, Q_COL, A_COL]

TEST_COLS = COLS + ["useless_tab"]

CORPUS_DICT = "corpus_dict.pkl"

def process_tokens(tokens: list[str]):
    """
    Returns a list of processed tokens, according to the specified options,
    without ponctuation.
    """
    
    wnl = WordNetLemmatizer()

    # lowercased words without punctuation
    ret = [word.lower() for word in tokens if word.isalpha()]
    # lemmatized words that are not stopwords
    ret = [wnl.lemmatize(word) for word in ret if word not in stopwords.words()]
    
    return ret

def process_corpus_to_dict(corpus_path=None, data_frame=None):
    """
    Pre-processes a corpus and returns a dictionary whose keys are the
    CATEGORIES (see utils.py) and whose values are sets of pre-processed
    tokens.
    """

    # --- Auxiliary Functions (pretty code, yay) ---

    def add_tokens_to_dict(d, row, tokens):
        label = row[LBL_COL]
        d[label] = d[label].union(tokens)

    # --- Actual function ---

    assert corpus_path is not None or data_frame is not None

    if (data_frame is None):
        corpus = pd.read_csv(corpus_path, sep = "\t", names = COLS)
    else:
        corpus = data_frame

    n_samples = corpus.shape[0]
    dic = dict((key, set()) for key in CATEGORIES)

    print("-- Beginning corpus_to_dict --")
    for index, row in corpus.iterrows():
        if round(index / n_samples, 2) == 0.25:
            print("-- 25% --")
        elif round(index / n_samples, 2) == 0.50:
            print("-- 50% --")
        elif round(index / n_samples, 2) == 0.75:
            print("-- 75% --")

        tokens = word_tokenize(' '.join([row[Q_COL], row[A_COL]]))

        processed_tokens = process_tokens(tokens)

        add_tokens_to_dict(dic, row, processed_tokens)

    return dic

def analyze_data(corpus: pd.DataFrame):
	geography_data = corpus[corpus[LBL_COL] == "GEOGRAPHY"]
	music_data = corpus[corpus[LBL_COL] == "MUSIC"]
	lit_data = corpus[corpus[LBL_COL] == "LITERATURE"]
	history_data = corpus[corpus[LBL_COL] == "HISTORY"]
	sci_data = corpus[corpus[LBL_COL] == "SCIENCE"]

	plt.bar(CATEGORIES, [geography_data.shape[0], music_data.shape[0], lit_data.shape[0], history_data.shape[0], sci_data.shape[0]])
	plt.show()

def save_object(obj, filename):
	with open(filename, 'wb') as out:
		pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
	with open(filename, 'rb') as inp:
		return pickle.load(inp)
