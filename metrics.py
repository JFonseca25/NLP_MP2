from nltk.corpus.reader import util
from numpy import array, where, zeros, float64
from pandas import DataFrame, Index
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import utils as utils

y_true = array(['MUSIC', 'LITERATURE', 'LITERATURE', 'MUSIC', 'MUSIC', 'SCIENCE'])
y_pred = array(['HISTORY', 'SCIENCE', 'SCIENCE', 'SCIENCE', 'MUSIC', 'SCIENCE'])

METRICS = ["Accuracy, Kappa"]

# confusion_matrix(y_true, y_pred, labels=CATEGORIES)


def result_table(y_true, y_pred):
	"""Build a result table with pretended metrics for a given model.
		- 
		- y_pred: list with models' y_pred
	"""
	num_models, category_size = len(y_pred), len(utils.CATEGORIES) + 1 # +1: GENERAL
	accuracy, kappas = [], zeros(shape=(num_models,num_models), 
								dtype=float64)
	

	for model in y_pred:
		accuracy += [complete_accuracy(y_true, model[1])]
	
	accuracy = array(accuracy)

	#print("Accuracy ", accuracy)

	for first in range(num_models-1):
		for second in range(first + 1, num_models):
			kappa = float64(cohen_kappa(y_pred[first][1], 
							y_pred[second][1]))
			kappas[first][second] = kappa
			kappas[second][first] = kappa
		kappas[first][first] = 1 # every model agrees with itself
			
	
	#for model in range(num_models):
	#	print(y_pred[model][0], " Accuracy:\n", accuracy[model])

	models = [model[0] for model in y_pred]
	
	categories = utils.CATEGORIES + ["GENERAL"]
	print("Accuracy Dataframe:\n", DataFrame(accuracy, 
						index=Index(models), 
						columns=categories))

	#print("Kappas:\n" + str(kappas))
	print("Kappa Dataframe:\n", DataFrame(kappas, 
						index=Index(models), 
						columns=models))

	return accuracy, kappas
	#return DataFrame([category[1] for category in accuracy], index=Index(CATEGORIES+["GENERAL"]), columns=METRICS).T

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
	#accuracies = []
	#for category in utils.CATEGORIES:
	#	accuracies += [(category, category_accuracy(y_true, y_pred, category))]
	#accuracies += [("GENERAL", accuracy_score(y_true, y_pred))]
	#return accuracies

	# WIP: already working for a given model
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
