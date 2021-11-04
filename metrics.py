from numpy import array, where
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
	accuracy, kappas = [], []
	
	for model in y_pred:
		print(model[0], complete_accuracy(y_true, model[1]))
		accuracy += [complete_accuracy(y_true, model[1])]
	for first in range(len(y_pred)-1):
		for second in range(first+1, len(y_pred)):
			kappas += [cohen_kappa(y_pred[first][1], y_pred[second][1])]

	print("Accuracy:")
	for model in y_pred:
		print(model[0], accuracy)

	print("Kappas: " + str(kappas))
	#return DataFrame([category[1] for category in accuracy], index=Index(CATEGORIES+["GENERAL"]), columns=METRICS).T

def get_category_entries(arr, category):
	return where(arr == category)[0]

def select_lines(y_true, y_pred):
	"""Select wanted lines from both y_true and y_pred."""
	general_y_true = array(general_y_true)
	general_y_pred = array(general_y_pred)
	category_true_ix = get_category_entries(general_y_true, category)

	# return both 


def category_accuracy(general_y_true, general_y_pred, category):
	general_y_true = array(general_y_true)
	general_y_pred = array(general_y_pred)
	category_true_ix = get_category_entries(general_y_true, category)

	if category_true_ix.size == 0:
		return "-" # No data in y_true.
	else:
		return accuracy_score(general_y_true[category_true_ix], general_y_pred[category_true_ix])
		
def complete_accuracy(y_true, y_pred):
	accuracies = []
	for category in utils.CATEGORIES:
		accuracies += [(category, category_accuracy(y_true, y_pred, category))]
	accuracies += [("GENERAL", accuracy_score(y_true, y_pred))]
	return accuracies


def cohen_kappa(y_model1, y_model2):
	y_model1 = array(y_model1)
	y_model2 = array(y_model2)
	return cohen_kappa_score(y_model1, y_model2)
