import pandas as pd

from utils import *
from .model import Model
from sklearn.feature_extraction.text import CountVectorizer

class EModel(Model):
    def __init__(self, training_set: pd.DataFrame = None, wd_mat = None) -> None:
        super().__init__(training_set)
        self.model = None
        self.wd_mat = wd_mat
        self.doc_list = None

    def get_doc_list(self):
        if self.doc_list != None:
            return self.doc_list

        docs = []
        for cat in CATEGORIES:
            cat_string = ' '.join([word for word in self.training_set[cat]])
            docs.append(cat_string)
            
        self.doc_list = docs

        return docs

    def vectorize_doc(self, doc):
        wnl = WordNetLemmatizer()
        lemmatized_doc = ' '.join([wnl.lemmatize(word) for word in doc])
        cv = CountVectorizer(stop_words='english')

        return cv.fit_transform(lemmatized_doc).toarray()

    def vectorize_doc_list(self):
        """
        Vectorizes self.doc_list into a count vector.
        """

        if self.doc_list == None:
            self.doc_list = self.get_doc_list()

        cv = CountVectorizer()
        return cv.fit_transform(self.get_doc_list()).toarray()

    def get_wd_mat(self):
        """
        Returns a wd-matrix with the class' vectorization method.
        """

        if self.wd_mat != None:
            return self.wd_mat

        self.wd_mat = self.vectorize_doc_list()

        return self.wd_mat

    def train(self):
        """
        Trains the model.
        """
        raise NotImplementedError("Training not implemented for class " + self.__class__.__name__)

    def test_row(self, row):
        raise NotImplementedError("test_row not implemented in class " + __class__.__name__)
