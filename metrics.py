from nltk.corpus.reader import util
from numpy import array, where, zeros, float64
from pandas import DataFrame, Index
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import utils as utils

y_true = array(['MUSIC', 'LITERATURE', 'LITERATURE', 'MUSIC', 'MUSIC', 'SCIENCE'])
y_pred = array(['HISTORY', 'SCIENCE', 'SCIENCE', 'SCIENCE', 'MUSIC', 'SCIENCE'])

METRICS = ["Accuracy, Kappa"]

# confusion_matrix(y_true, y_pred, labels=CATEGORIES)


def results(y_true, y_pred):
	"""Calculate models' results.
		- 
		- y_pred: list with models' y_pred
	"""
	num_models = len(y_pred)
	accuracy, kappas = [], zeros(shape=(num_models,num_models), 
								dtype=float64)
	

	for model in y_pred:
		accuracy += [complete_accuracy(y_true, model[1])]
	
	accuracy = array(accuracy)

	for first in range(num_models-1):
		for second in range(first + 1, num_models):
			kappa = float64(cohen_kappa(y_pred[first][1], 
							y_pred[second][1]))
			kappas[first][second] = kappa
			kappas[second][first] = kappa
		kappas[first][first] = 1 # every model agrees with itself
				
	return accuracy, kappas


def result_table(accuracy, kappas, models):
	
	categories = utils.CATEGORIES + ["GENERAL"]
	accuracy_df = DataFrame(accuracy, index=Index(models), 
							columns=categories)
	kappas_df = DataFrame(kappas, index=Index(models), columns=models)

	print("Accuracy Dataframe:\n", accuracy_df)
	print("Kappa Dataframe:\n", kappas_df)
	return accuracy_df, kappas_df


def get_category_entries(arr, category):
	return where(arr == category)[0]


def category_accuracy(general_y_true, general_y_pred, category):
	general_y_true = array(general_y_true)
	general_y_pred = array(general_y_pred)
	category_true_ix = get_category_entries(general_y_true, category)

	if category_true_ix.size == 0:
		return float64(1) # No data in y_true.
	else:
		return accuracy_score(general_y_true[category_true_ix], 
					general_y_pred[category_true_ix])
		
def complete_accuracy(y_true, y_pred):
	"""Determines accuracy for all categories + general for a given model."""
	category_size = len(utils.CATEGORIES)
	accuracies = zeros(category_size + 1) # +1: GENERAL
	for category in range(category_size):
		accuracies[category] = category_accuracy(y_true, y_pred, 
						utils.CATEGORIES[category])
	accuracies[category_size] = accuracy_score(y_true, y_pred)
	return accuracies


def cohen_kappa(y_model1, y_model2):
	y_model1 = array(y_model1)
	y_model2 = array(y_model2)
	return cohen_kappa_score(y_model1, y_model2)
