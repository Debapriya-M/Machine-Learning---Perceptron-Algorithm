import argparse
import numpy as np
import pandas as pd
import random

max_iter = 100000

def getTrainingInputs(arguments):
	# print("Inside here --> 1")
	# data = np.loadtxt(arguments.dataset, dtype=str, delimiter=',')
	dataframe = pd.read_csv(arguments.dataset_path)
	# l = dataframe.iloc[:, -1].values
	labels = np.where(dataframe.iloc[:, -1].values == 0, -1, 1)

	instances = dataframe.iloc[:, :-1]
	instances['bias'] = 1
	instances = instances.values
	return instances, labels

def train(instances, labels):
	weight = np.zeros(instances.shape[1],)
	count = 0
	while count < max_iter:
		product = labels * np.dot(instances, weight)
		if np.any(product <= 0):
			temp = np.where(product <= 0)[0][0]
			weight = weight + labels[temp] * instances[temp]
			count += 1
		else:
			break
	return weight

def calculateError(instances, labels, weight):
	term = len(np.where(labels * np.dot(instances, weight) <= 0)[0])
	error = term / instances.shape[0]
	print('Error: ', error)
	return error

def getPrediction(inputs, wts, bias):
	epoch_op = np.dot(inputs, wts) + bias
	# Generating the Activation Function and returning the activating function for epoch_output
	if epoch_op > 0 :
		return 1
	else:
		return 0

def computeLoss(label, predicted_label):
	# Calculating the 0-1 Loss here and returning the loss
	if label==predicted_label:
		return 1
	else:
		return 0

# Calculation the Empirical Risk/Loss Minimization for the both the datasets here --->
def calculateERM(instances, labels):
	# weight = train(instances, labels)
	# error = calculateError(instances, labels, weight)
	count = 0
	epochs = 10
	learning_rate = 0.0001
	weights = np.random.rand(len(instances[0]))
	bias = np.zeros(len(labels))
	for epoch in range(epochs):
		row = 0
		init_loss = 0
		print ('Epoch Number: ', epoch + 1)
		for train_inputs, label in zip(instances, labels):
			predicted_label = getPrediction(train_inputs, weights, bias[row])
			numpyInstances = np.asarray(train_inputs)
			weights += learning_rate * (label - predicted_label) * numpyInstances
			bias[row] += learning_rate * (label - predicted_label)
			init_loss += computeLoss(label, predicted_label)
			row += 1
		
		error = init_loss / len(instances)

	print ('ERM Loss: ', error)
	print ('Weights: ', weights)
	print ('Accuracy: ', 1 - error)

def calculateCrossValidation(instances, label):
	batches = []
	k_folds = 10
	index, d = instances.shape
	foldsize = index//k_folds
	s = foldsize + (1 if index % k_folds != 0 else 0)
	indexes = list(range(instances.shape[0]))
	random.shuffle(indexes)
	instances = instances[indexes]
	label = label[indexes]

	for i in range(k_folds):
		start, end = s * i, s * (i + 1)
		batches.append((instances[start:end], label[start:end]))

	weights, errors = [], []
	for i in range(k_folds):
		print('Executing Fold #: %d' % (i + 1))
		train_X, train_y, test_X, test_y = None, None, None, None
		for j, (instances, label) in enumerate(batches):
			if j == i:
				test_X, test_y = instances, label
			else:
				if train_X is None:
					train_X, train_y = instances, label
				else:
					train_X, train_y = np.append(train_X, instances, axis=0), np.append(train_y, label, axis=0)
		weights.append(train(train_X, train_y))
		errors.append(calculateError(test_X, test_y, weights[-1]))
		print('Weight: %s \nError: %s' % (weights[-1], errors[-1]))
	mean_error = np.mean(errors)
	print('Errors: %s \nMean Error: %s' % (errors, mean_error))
	print ('Accuracy: ', 1 - mean_error)	

def main():
	parser = argparse.ArgumentParser(description='Implementation of Perceptron')
	parser.add_argument('--dataset', dest = 'dataset_path', action = 'store', type = str, help='location of dataset')
	parser.add_argument('--mode', dest='mode', action='store', type=str, default='erm', help='mode of the algorithm - erm or kfold')
	arguments = parser.parse_args()
	# reading the csv file and extract X set and bias
	instances, labels = getTrainingInputs(arguments)
	
	if arguments.mode == 'erm':
		calculateERM(instances, labels)
	elif arguments.mode == 'cv':
		calculateCrossValidation(instances, labels)
	else:
		print('Mode of algorithmis is incorrect. Please use "erm" or "cv".')


if __name__ == '__main__':
	main()



