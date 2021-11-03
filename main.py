from os import pipe
from nltk.data import load
import pandas as pd
import os
import pickle
from utils import *
from dice import Dice
from jaccard import Jaccard
from naive_bayes import NaiveBayes
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

def save_object(obj, filename):
	with open(filename, 'wb') as out:
		pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
	with open(filename, 'rb') as inp:
		return pickle.load(inp)

def main():
	corpus = pd.read_table(TRAINING_FILE, sep="\t", names=COLS, converters={Q_COL: str, A_COL: str})
	test_ds = pd.read_table(TESTING_FILE, sep="\t", names=TEST_COLS, converters={Q_COL: str, A_COL: str})

	if not os.path.isfile(CORPUS_DICT):
		print("--- Processing corpus for BoW Models ---")
		bow = process_corpus_to_dict(data_frame=corpus)
		save_object(bow, CORPUS_DICT)
	else:
		bow = load_object(CORPUS_DICT)

	true_labels = test_ds[LBL_COL].to_list()

	jaccard_model = Jaccard(bow)
	dice_model = Dice(bow)
	nb_model_cv = NaiveBayes(corpus, vectorizer=CountVectorizer(stop_words='english'))
	nb_model_tfidf = NaiveBayes(corpus)

	print("--- Currently training learning models ---")
	nb_model_cv.train()
	print("Naive Bayes with CountVec done!")
	nb_model_tfidf.train()
	print("Naive Bayes with tf-idf done!")

	print("--- Testing ---")

	print("Jaccard -> ", accuracy_score(true_labels, jaccard_model.test(test_ds)))
	print("Dice -> ", accuracy_score(true_labels, dice_model.test(test_ds)))
	print("NB CV -> ", accuracy_score(true_labels, nb_model_cv.predict(test_ds)))
	print("NB tf-idf -> ", accuracy_score(true_labels, nb_model_tfidf.predict(test_ds)))

if __name__ == '__main__':
	main()