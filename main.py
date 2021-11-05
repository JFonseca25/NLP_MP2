import os

from sklearn import svm
from lr import LR
import metrics as mt
import numpy as np
import pandas as pd

from knn import KNN
from perceptron import PerceptronClassifier
from rfc import RForest
from sgd import SGD
from svm import SVM
from utils import *
from dice import Dice
from jaccard import Jaccard
from naive_bayes import NaiveBayes
from lemma_tokenizer import LemmaTokenizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def results_graph(results: list, model_names: list):
	plt.bar(model_names, results)
	plt.show()

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

	#jaccard_model = Jaccard(bow)
	#dice_model = Dice(bow)
	nb_model_cv_1 = NaiveBayes(corpus, vectorizer=CountVectorizer())
	nb_model_cv_2 = NaiveBayes(corpus, vectorizer=CountVectorizer(stop_words='english'))
	nb_model_cv_3 = NaiveBayes(corpus, vectorizer=CountVectorizer(tokenizer=LemmaTokenizer()))
	
	nb_model_tfidf_1 = NaiveBayes(corpus, vectorizer=TfidfVectorizer())
	nb_model_tfidf_2 = NaiveBayes(corpus)
	nb_model_tfidf_3 = NaiveBayes(corpus, vectorizer=TfidfVectorizer(tokenizer=LemmaTokenizer()))

	svm_model_tfidf_1 = SVM(corpus, vectorizer=TfidfVectorizer(tokenizer=LemmaTokenizer()))
	svm_model_tfidf_2 = SVM(corpus, model=svm.LinearSVC(), vectorizer=TfidfVectorizer(tokenizer=LemmaTokenizer()))
	svm_model_tfidf_3 = SVM(corpus, model=svm.SVC(decision_function_shape='ovo'), vectorizer=TfidfVectorizer(tokenizer=LemmaTokenizer()))

	knn_model_tfidf_1 = KNN(corpus, vectorizer=TfidfVectorizer())
	knn_model_tfidf_2 = KNN(corpus)
	knn_model_tfidf_3 = KNN(corpus, vectorizer=TfidfVectorizer(tokenizer=LemmaTokenizer()))

	sgd_model = SGD(corpus, vectorizer=TfidfVectorizer(tokenizer=LemmaTokenizer()))

	lr_model = LR(corpus, vectorizer=TfidfVectorizer(tokenizer=LemmaTokenizer()))

	perc_model = PerceptronClassifier(corpus, vectorizer=TfidfVectorizer(tokenizer=LemmaTokenizer()))

	rf_model = RForest(corpus, vectorizer=TfidfVectorizer(tokenizer=LemmaTokenizer()))

	print("--- Currently training learning models ---")
	nb_model_cv_1.train()
	nb_model_cv_2.train()
	nb_model_cv_3.train()
	print("Naive Bayes with CountVectorizer done!")
	nb_model_tfidf_1.train()
	nb_model_tfidf_2.train()
	nb_model_tfidf_3.train()
	print("Naive Bayes with tf-idf done!")
	knn_model_tfidf_1.train()
	knn_model_tfidf_2.train()
	knn_model_tfidf_3.train()
	print("KNN with tf-idf done!")
	sgd_model.train()
	print("SGD with tf-idf done!")
	lr_model.train()
	print("LR with tf-idf done!")
	perc_model.train()
	print("Perceptron with tf-idf done!")
	rf_model.train()
	print("Random Forest with tf-idf done!")
	svm_model_tfidf_1.train()
	svm_model_tfidf_2.train()
	svm_model_tfidf_3.train()
	print("SVM with tf-idf done!")

	print("--- Testing ---")

	#jaccard_y = jaccard_model.test(test_ds)
	#print("Jaccard testing done.")
	
	nb_cv_1_y = nb_model_cv_1.predict(test_ds)
	nb_cv_2_y = nb_model_cv_2.predict(test_ds)
	nb_cv_3_y = nb_model_cv_3.predict(test_ds)

	nb_tfidf_1_y = nb_model_tfidf_1.predict(test_ds)
	nb_tfidf_2_y = nb_model_tfidf_2.predict(test_ds)
	nb_tfidf_3_y = nb_model_tfidf_3.predict(test_ds)

	knn_tfidf_1_y = knn_model_tfidf_1.predict(test_ds)
	knn_tfidf_2_y = knn_model_tfidf_2.predict(test_ds)
	knn_tfidf_3_y = knn_model_tfidf_3.predict(test_ds)

	sgd_model_y = sgd_model.predict(test_ds)

	lr_model_y = lr_model.predict(test_ds)

	perc_y = perc_model.predict(test_ds)

	rforest_y = rf_model.predict(test_ds)

	svm_tfidf_1_y = svm_model_tfidf_1.predict(test_ds)
	svm_tfidf_2_y = svm_model_tfidf_2.predict(test_ds)
	svm_tfidf_3_y = svm_model_tfidf_3.predict(test_ds)

	#print("Jaccard -> ", accuracy_score(true_labels, jaccard_y))
	#print("Dice -> ", accuracy_score(true_labels, dice_y))
	print("- NB CV 1 -\n", classification_report(true_labels, nb_cv_1_y))
	print("- NB CV 2 -\n", classification_report(true_labels, nb_cv_2_y))
	print("- NB CV 3 -\n", classification_report(true_labels, nb_cv_3_y))

	print("- NB tf-idf 1 -\n", classification_report(true_labels, nb_tfidf_1_y))
	print("- NB tf-idf 2 -\n", classification_report(true_labels, nb_tfidf_2_y))
	print("- NB tf-idf 3 -\n", classification_report(true_labels, nb_tfidf_3_y))

	print("- KNN tf-idf 1 -\n", classification_report(true_labels, knn_tfidf_1_y))
	print("- KNN tf-idf 2 -\n", classification_report(true_labels, knn_tfidf_2_y))
	print("- KNN tf-idf 3 -\n", classification_report(true_labels, knn_tfidf_3_y))

	print("- SGD tf-idf -\n", classification_report(true_labels, sgd_model_y))

	print(" - LR tf-idf -\n", classification_report(true_labels, lr_model_y))
	
	print(" - Perceptron tf-idf -\n", classification_report(true_labels, perc_y))

	print(" - Random Forest tf-idf -\n", classification_report(true_labels, rforest_y))

	print(" - Random Forest tf-idf -\n", classification_report(true_labels, rforest_y))
	print(" - Random Forest tf-idf -\n", classification_report(true_labels, rforest_y))
	print(" - Random Forest tf-idf -\n", classification_report(true_labels, rforest_y))

	print("- SVM tf-idf 1 -\n", classification_report(true_labels, svm_tfidf_1_y))
	print("- SVM (linear) tf-idf 2 -\n", classification_report(true_labels, svm_tfidf_2_y))
	print("- SVM (ovo) tf-idf 3 -\n", classification_report(true_labels, svm_tfidf_3_y))

	results = [accuracy_score(true_labels, i) for i in [nb_cv_1_y, nb_cv_2_y,
				nb_cv_3_y, nb_tfidf_1_y, nb_tfidf_2_y, nb_tfidf_3_y, 
					knn_tfidf_1_y, knn_tfidf_2_y, knn_tfidf_3_y, sgd_model_y, 
						lr_model_y, perc_y, rforest_y, svm_tfidf_1_y, svm_tfidf_2_y,
							svm_tfidf_3_y]]
	
	print("--- MAX ACCURACY: {} ---".format(max(results)))

	model_names = ["NB CV 1", "NB CV 2", "NB CV 3", "NB tf-idf 1", "NB tf-idf 2",
					"NB tf-idf 3", "KNN tf-idf 1", "KNN tf-idf 2", "KNN tf-idf 3",
						"SGD tf-idf", "LR tf-idf", "Perceptron tf-idf", "RF tf-idf",
							"SVM tf-idf 1", "SVM (linear) tf-idf 2", "SVM (ovo) tf-idf 3"]

	results_graph(np.array(results), np.array(model_names))


	#mt.result_table(true_labels, [("Jaccard", jaccard_y), ("Naive Bayes", nb_cv_y)]) WIP

if __name__ == "__main__":
	main()
