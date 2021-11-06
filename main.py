from operator import index
from matplotlib.pyplot import xlim
from nltk.corpus.reader import aligned
from numpy.core.numeric import cross
from sklearn import svm
import pandas as pd
import numpy as np

from knn import KNN
from sklearn import svm
from utils import *
from lemma_tokenizer import LemmaTokenizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate

def get_corpus(training_set):
        corpus = []
        for ix, row in training_set.iterrows():
            string = ' '.join([row[Q_COL], row[A_COL]])
            corpus.append(string)
        return corpus

def tokenizer(s):
	ret = [t.lower() for t in word_tokenize(s) if t.isalpha()]
	ret = [WordNetLemmatizer().lemmatize(t) for t in ret]
	return ret

def print_cross_val_results(score):
	print("\n-- Results --")
	print("- Avg. balanced accuracy:", score["test_balanced_accuracy"].mean())
	print("- Std. Dev. of balanced accuracy:", score["test_balanced_accuracy"].std())

	print("- Avg. f1_weighted:", score["test_f1_weighted"].mean())
	print("- Std. Dev. of f1_weighted:", score["test_f1_weighted"].std(), '\n')

def plot_cross_val_results(title, labels, mean, stddev):	
	df = pd.DataFrame(np.c_[mean, stddev], index=labels)

	df.plot(kind='bar')
	plt.grid(b=True, which='both', axis='y', linestyle='-', linewidth=0.5)
	plt.legend(["Average", "Std. Deviation"])
	plt.hlines(max(mean), linestyles='dashed', xmin=-0.25, xmax=2, colors='grey')
	plt.hlines(min(stddev), linestyles='dashed', xmin=0, xmax=2.25, colors='grey')
	plt.xticks(rotation='horizontal')
	plt.title(title)
	plt.show()

def clear_result_lists(l1, l2, l3, l4):
	l1.clear()
	l2.clear()
	l3.clear()
	l4.clear()

def append_results(mean_acc, std_acc, mean_f1, std_f1, score):
	mean_acc.append(score['test_balanced_accuracy'].mean())
	std_acc.append(score['test_balanced_accuracy'].std())

	mean_f1.append(score['test_f1_weighted'].mean())
	std_f1.append(score['test_f1_weighted'].std())

def cross_val_model(model, vectorizer, folds=20):
	whole_corpus = pd.read_table("whole_corpus.txt", sep="\t", names=COLS, converters={Q_COL: str, A_COL: str})

	X = get_corpus(whole_corpus)
	y = whole_corpus[LBL_COL]

	X = vectorizer.fit_transform(X)

	scoring = ['balanced_accuracy', 'f1_weighted']

	print("-- Beginning cross val. with {} folds --".format(folds))

	return cross_validate(model, X, y, scoring=scoring, cv=folds)

def main():
	mean_acc = []
	std_acc = []
	mean_f1 = []
	std_f1 = []

	labels = ["lc", "lc + sw_rem", "lc + sw_rem + lemm"]

	print("---- LSVM ----\n")
	print("--- Cross Val. of LSVM CV with lowercasing -- ")
	score = cross_val_model(svm.LinearSVC(), CountVectorizer())
	print_cross_val_results(score)
	append_results(mean_acc, std_acc, mean_f1, std_f1, score)
	print("--- Cross Val. of LSVM CV with lowercasing and sw_removal -- ")
	score = cross_val_model(svm.LinearSVC(), CountVectorizer(stop_words='english'))
	print_cross_val_results(score)
	append_results(mean_acc, std_acc, mean_f1, std_f1, score)

	print("--- Cross Val. of LSVM CV with lemmatization, sw_removal removal, lowercasing --")
	score = cross_val_model(svm.LinearSVC(), CountVectorizer(tokenizer=tokenizer))
	print_cross_val_results(score)
	append_results(mean_acc, std_acc, mean_f1, std_f1, score)

	plot_cross_val_results("Weighted Accuracy Mean / Std. Dev for LSVM with CountVectorizer", labels, mean_acc, std_acc)
	plot_cross_val_results("Weighted F1-Measure Mean / Std. Dev for LSVM with CountVectorizer", labels, mean_f1, std_f1)
	clear_result_lists(mean_acc, std_acc, mean_f1, std_f1)

	print("--- Cross Val. of LSVM tf-idf with lowercasing -- ")
	score = cross_val_model(svm.LinearSVC(), TfidfVectorizer())
	print_cross_val_results(score)
	append_results(mean_acc, std_acc, mean_f1, std_f1, score)

	print("--- Cross Val. of LSVM tf-idf with lowercasing and sw_removal -- ")
	score = cross_val_model(svm.LinearSVC(), TfidfVectorizer(stop_words='english'))
	print_cross_val_results(score)
	append_results(mean_acc, std_acc, mean_f1, std_f1, score)

	print("--- Cross Val. of LSVM tf-idf with lemmatization, sw_removal removal, lowercasing --")
	score = cross_val_model(svm.LinearSVC(), TfidfVectorizer(tokenizer=tokenizer))
	print_cross_val_results(score)
	append_results(mean_acc, std_acc, mean_f1, std_f1, score)

	plot_cross_val_results("Weighted Accuracy Mean / Std. Dev for LSVM with TfidfVectorizer", labels, mean_acc, std_acc)
	plot_cross_val_results("Weighted F1-Measure Mean / Std. Dev for LSVM with TfidfVectorizer", labels, mean_f1, std_f1)
	clear_result_lists(mean_acc, std_acc, mean_f1, std_f1)

	print("---- NB ----\n")

	print("--- Cross Val. of NB CV with lowercasing --")
	score = cross_val_model(MultinomialNB(), CountVectorizer())
	print_cross_val_results(score)
	append_results(mean_acc, std_acc, mean_f1, std_f1, score)

	print("--- Cross Val. of NB CV with lowercasing and sw_removal --")
	score = cross_val_model(MultinomialNB(), CountVectorizer(stop_words='english'))
	print_cross_val_results(score)
	append_results(mean_acc, std_acc, mean_f1, std_f1, score)

	print("--- Cross Val. of NB CV with lemmatization, sw_removal removal, lowercasing --")
	score = cross_val_model(MultinomialNB(), CountVectorizer(tokenizer=tokenizer))
	print_cross_val_results(score)
	append_results(mean_acc, std_acc, mean_f1, std_f1, score)

	plot_cross_val_results("Weighted Accuracy Mean / Std. Dev for NB with CountVectorizer", labels, mean_acc, std_acc)
	plot_cross_val_results("Weighted F1-Measure Mean / Std. Dev for NB with CountVectorizer", labels, mean_f1, std_f1)
	clear_result_lists(mean_acc, std_acc, mean_f1, std_f1)

	print("--- Cross Val. of NB tf-idf with lowercasing --")
	score = cross_val_model(MultinomialNB(), TfidfVectorizer())
	print_cross_val_results(score)
	append_results(mean_acc, std_acc, mean_f1, std_f1, score)

	print("--- Cross Val. of NB tf-idf with lowercasing and sw_removal --")
	score = cross_val_model(MultinomialNB(), TfidfVectorizer(stop_words='english'))
	print_cross_val_results(score)
	append_results(mean_acc, std_acc, mean_f1, std_f1, score)

	print("--- Cross Val. of NB tf-idf with lemmatization, sw_removal removal, lowercasing --")
	score = cross_val_model(MultinomialNB(), TfidfVectorizer(tokenizer=tokenizer))
	print_cross_val_results(score)
	append_results(mean_acc, std_acc, mean_f1, std_f1, score)

	plot_cross_val_results("Weighted Accuracy Mean / Std. Dev for NB with TfidfVectorizer", labels, mean_acc, std_acc)
	plot_cross_val_results("Weighted F1-Measure Mean / Std. Dev for NB with TfidfVectorizer", labels, mean_f1, std_f1)

if __name__ == "__main__":
	main()
