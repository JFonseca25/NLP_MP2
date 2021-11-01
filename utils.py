import nltk

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
LBL_ROW = "l"

"""Label for the Question Row of the training and testing set"""
Q_ROW = "q"

"""Label for the Answer Row of the training and testing sets"""
A_ROW = "a"

STOPWORDS = set(stopwords.words('english'))

TRAINING_ROWS = [LBL_ROW, Q_ROW, A_ROW]

TESTING_ROWS = [Q_ROW, A_ROW]
