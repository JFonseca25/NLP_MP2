from utils import *

class LemmaTokenizer:
    """
    Class passed to Scikit-Learn tokenizers to tokenize a set of documents,
    lemmatizing those tokens and removing english stopwords.
    """

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        ret = [t.lower() for t in word_tokenize(doc) if t.isalpha()]
        return [self.wnl.lemmatize(t) for t in ret if t not in STOPWORDS]
