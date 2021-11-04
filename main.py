import os
import metrics as mt
import pandas as pd

from knn import KNN
from utils import *
from dice import Dice
from jaccard import Jaccard
from naive_bayes import NaiveBayes
from lemma_tokenizer import LemmaTokenizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
	#dice_model = Dice(bow)
	nb_model_cv_1 = NaiveBayes(corpus, vectorizer=CountVectorizer(stop_words='english'))
	nb_model_cv_2 = NaiveBayes(corpus, vectorizer=CountVectorizer(tokenizer=LemmaTokenizer()))
	
	nb_model_tfidf_1 = NaiveBayes(corpus)
	nb_model_tfidf_2 = NaiveBayes(corpus, vectorizer=TfidfVectorizer(tokenizer=LemmaTokenizer()))

	knn_model_cv_1 = KNN(corpus, vectorizer=CountVectorizer(stop_words='english'))
	knn_model_cv_2 = KNN(corpus, vectorizer=CountVectorizer(tokenizer=LemmaTokenizer()))

	knn_model_tfidf_1 = KNN(corpus)
	knn_model_tfidf_2 = KNN(corpus, vectorizer=TfidfVectorizer(tokenizer=LemmaTokenizer()))

	print("--- Currently training learning models ---")
	nb_model_cv_1.train()
	nb_model_cv_2.train()
	print("Naive Bayes with CountVectorizer done!")
	nb_model_tfidf_1.train()
	nb_model_tfidf_2.train()
	print("Naive Bayes with tf-idf done!")
	knn_model_cv_1.train()
	knn_model_cv_2.train()
	print("KNN with CountVectorizer done!")
	knn_model_tfidf_1.train()
	knn_model_tfidf_2.train()
	print("KNN with tf-idf done!")

	print("--- Testing ---")

	jaccard_y = jaccard_model.test(test_ds)
	#print("Jaccard testing done.")
	
	nb_cv_1_y = nb_model_cv_1.predict(test_ds)
	nb_cv_2_y = nb_model_cv_2.predict(test_ds)

	nb_tfidf_1_y = nb_model_tfidf_1.predict(test_ds)
	nb_tfidf_2_y = nb_model_tfidf_2.predict(test_ds)

	knn_cv_1_y = knn_model_cv_1.predict(test_ds)
	knn_cv_2_y = knn_model_cv_2.predict(test_ds)

	knn_tfidf_1_y = knn_model_tfidf_1.predict(test_ds)
	knn_tfidf_2_y = knn_model_tfidf_2.predict(test_ds)

	#print(mt.complete_accuracy(true_labels, jaccard_y))
	#print(mt.complete_accuracy(true_labels, nb_cv_1_y))
	#print("Kappa: " + str(mt.cohen_kappa(jaccard_y, nb_cv_1_y)))

	print("Jaccard -> ", accuracy_score(true_labels, jaccard_y))
	#print("Dice -> ", accuracy_score(true_labels, dice_y))
	print("NB CV -> ", accuracy_score(true_labels, nb_cv_1_y))
	print("NB CV 2 (lemmatization) -> ", accuracy_score(true_labels, nb_cv_2_y))

	print("NB tf-idf 1 -> ", accuracy_score(true_labels, nb_tfidf_1_y))
	print("NB tf-idf 2 (lemmatization) -> ", accuracy_score(true_labels, nb_tfidf_2_y))

	print("KNN CV 1 -> ", accuracy_score(true_labels, knn_cv_1_y))
	print("KNN CV 2 (lemmatization) -> ", accuracy_score(true_labels, knn_cv_2_y))

	print("KNN tf-idf 1 -> ", accuracy_score(true_labels, knn_tfidf_1_y))
	print("KNN tf-idf 2 (lemmatization) -> ", accuracy_score(true_labels, knn_tfidf_2_y))

	#mt.result_table(true_labels, [("Jaccard", jaccard_y), ("Naive Bayes", nb_cv_y)]) WIP

if __name__ == "__main__":
	main()
