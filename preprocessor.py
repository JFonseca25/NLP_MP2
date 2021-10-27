import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

TRAINING_FILE = "trainWithoutDev.txt"
"""
    The names of the columns of the data table. Since we have some lines with
    more than one label, we need four.

    'lq' -> label or question
    'qa' -> question or answer
    'aeps' -> answer or epsilon (empty)
"""
COLUMN_NAMES = ["label", "lq", "qa", "aeps"]

CATEGORIES = ["GEOGRAPHY", "MUSIC", "LITERATURE", "HISTORY", "SCIENCE"]

STOPWORDS = set(stopwords.words('english'))

class Preprocessor():
    def __init__(self, training_file=TRAINING_FILE) -> None:
        self.data_table = pd.read_csv(training_file, sep="\t", names=COLUMN_NAMES)
        self.data_sets = dict((key, set()) for key in CATEGORIES)

    def process_corpus(self):
        
        # --- Auxiliary Functions (pretty code, yay) ---

        def get_question_and_answer():
            question = answer = ""

            if row["lq"] in CATEGORIES:
                question = row["qa"]
                answer = row["aeps"]
            else:
                question = row["lq"]
                answer = row["qa"]

            return question, answer

        def get_where_to_add():
            where_to_add = [row["label"]]
            
            if row["lq"] in CATEGORIES:
                where_to_add += [row["lq"]]

            return where_to_add

        def process_tokens(tokens):
            wnl = WordNetLemmatizer()

            tokens_without_sw = [word for word in tokens if word not in stopwords.words() and word.isalpha()]
            tokens_without_sw_lemmatized = [wnl.lemmatize(word) for word in tokens_without_sw]

            return tokens_without_sw_lemmatized

        def add_tokens_to_data_sets(tokens):
            for key in where_to_add:
                self.data_sets[key] = self.data_sets[key].union(tokens)

        # --- Actual function ---

        for index, row in self.data_table.iterrows():
            print(index)

            where_to_add = get_where_to_add()
            question, answer = get_question_and_answer()

            tokens = word_tokenize(question) + word_tokenize(answer)

            processed_tokens = process_tokens(tokens)

            add_tokens_to_data_sets(processed_tokens)
    
    def get_data_frame(self) -> pd.DataFrame:
        return self.data_table

    def get_data_sets(self) -> dict:
        return self.data_sets
