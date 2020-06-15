import argparse
import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt

def generate_kfold(instances, labels, folds):
	train = []
	index = np.arange(len(instances))
	firstFolds = len(instances)%folds
	foldsize = len(instances)//folds
	for fold_number in range(firstFolds):
		fold = []
		for fold_number in range(foldsize + 1):
			while True :
				temp = np.random.choice(index)
				if not any(temp in sublist for sublist in train):
					break
			fold.append(temp)
		train.append(fold)
	for fold_number in range(folds - firstFolds):
		fold = []
		for fold_number in range(foldsize):
			while True :
				temp = np.random.choice(index)
				if not any(temp in sublist for sublist in train):
					break
			fold.append(temp)
		train.append(fold)

	inputs = [[instances[i] for i in sublist] for sublist in train]
	outputs = [[labels[i] for i in sublist] for sublist in train]
	return inputs, outputs

def check(numerator, denominator):
	if denominator != 0:
		return (numerator/denominator)
	else:
		return 0

def calculateError(label, predicted_label):
	if label != predicted_label:
		return 1
	else:
		return 0

def generateDecisionStump(instances, labels, wts):
	threshold_val = -1
	erm_loss = sys.maxsize
	index = -1
	for col in range(len(instances[0])):
		loss = 0
		column = instances[:, col]
		temp_arr = []
		for val, label, weight in zip(column, labels, wts):
			temp_arr.append([val, label, weight])
		temp_arr.sort()
		for i in range(len(temp_arr)):
			if temp_arr[i][1]==1:
				loss += wts[i]
		if loss < erm_loss:
			erm_loss = loss
			threshold_val = temp_arr[0][0] - 1
			index = col

		for i in range(len(temp_arr)):
			loss = loss - (temp_arr[i][1] * temp_arr[i][2])

			if loss < erm_loss and temp_arr[i][0] != temp_arr[i+1][0] and i<len(temp_arr)-1:
				erm_loss = loss
				threshold_val = (temp_arr[i][0] + temp_arr[i+1][0])/2
				index = col

	return index, threshold_val

def getPrediction(learners, test_instances, test_labels):
	loss = 0
	for sample, label in zip(test_instances, test_labels):
		weightedSum = 0
		for l in learners:
			for index in l.keys():
				if sample[index] > l[index][0]:
					val = 1
				else:
					val = -1
				weightedSum += l[index][1] * val
		
		if np.sign(weightedSum) < 0:
			pred_label = -1
		else:
			pred_label = 1

		loss += calculateError(label, (pred_label))

	erm_loss = loss/len(test_instances)
	return erm_loss

def generateTrainableInput(instances, labels, epochs):
	arr_learners = []
	validation_loss = []
	ins = np.array(instances)
	wts = [1/len(ins)]* len(ins)
	for epoch in range(epochs):
		pred_labels_array = []
		error = 0
		col, threshold_val = generateDecisionStump(ins, labels, wts)
		
		for i in range(len(instances)):
			if ins[i, col] > threshold_val:
				pred_label = 1
			else:
				pred_label = -1
			pred_labels_array.append(pred_label)
			if labels[i]!=pred_label:
				error += wts[i]

		normalized_wts = 0
		l_wt = 0.5 * math.log( check(1 - error, error) + 0.00000001)
		temp1 = {col : [threshold_val, l_wt]}
		arr_learners.append(temp1)
		
		for i in range(len(instances)):
			term1 = l_wt * labels[i] * pred_labels_array[i]
			normalized_wts += wts[i] * math.exp(-term1)

		for i in range(len(instances)):
			term2 = l_wt * labels[i] * pred_labels_array[i]
			wts[i] = (wts[i] * math.exp(-term2) ) / normalized_wts
	return arr_learners, wts
	
	
def calculateERM(instances, labels, epochs):
	learners, weights = generateTrainableInput(instances, labels, epochs)
	erm_loss = getPrediction(learners, instances, labels)
	#Printing the values for ERM calculation of adaBoost algorithm
	print ('Weights: ', weights)
	print ('erm_loss: ', erm_loss)
	print ('Accuracy: ', 1 - erm_loss)

def calculateCrossValidation(instances, labels, epochs, mode):
	validation_loss = []
	total_erm_loss = []
	folds = 10
	for f in range(epochs):
		inputs, outputs = generate_kfold(instances, labels, folds)
		mean_fold_loss, erm_loss = 0, 0
		for i in range(folds):
			train_inputs, train_labels = [], []
			test_inputs, test_labels = inputs[i], outputs[i]
			
			for j in range(folds):
				if j!=i:
					train_inputs.extend(inputs[j])
					train_labels.extend(outputs[j])

			learners, weights = generateTrainableInput(train_inputs, train_labels, epochs)
			fold_loss = getPrediction(learners, test_inputs, test_labels)
			erm_loss += getPrediction(learners, train_inputs, train_labels)

			if mode != 'plot' and f == epochs - 1:
				print ('Fold ', i+1, ' Loss: ', fold_loss)
			
			mean_fold_loss += fold_loss

		total_erm_loss.append(erm_loss/folds)
		validation_loss.append(mean_fold_loss/folds)
		accuracy = mean_fold_loss/folds
	if mode!='plot':
		print ('Weights: ', weights)
		print ('Mean Fold Error: ', mean_fold_loss/folds)
		print ('Accuracy: ', 1 - accuracy)

	if mode=='plot':
		plot(validation_loss, total_erm_loss)

def getModeImplementation(instances, labels, mode):
	epochs = 10
	if mode == 'erm':
		calculateERM(instances, labels, epochs)
	elif mode == 'cv' or mode == 'plot':
		calculateCrossValidation(instances, labels, epochs, mode)
			
def plot(validation_loss, total_erm_loss):
	epochs = np.arange(len(validation_loss))
	plt.plot(epochs, validation_loss, marker = 'o', linestyle = ':', label = 'Validation Loss')
	plt.plot(epochs, total_erm_loss, marker = 'o', linestyle = ':', label = 'ERM loss')
	plt.title('ERM and Validation Error vs Number of rounds(' + str(len(epochs)) + ')')
	plt.xlabel('T: number of rounds in adaboost')
	plt.ylabel('Error')
	plt.legend()
	plt.show()

def main():
	parser = argparse.ArgumentParser(description='Adaptive Boosting - AdaBoost - Implemenentation')
	parser.add_argument('--dataset', dest='dataset_path', action='store', type=str, help='path to dataset')
	parser.add_argument('--mode', type=str, action='store', help='mode of algorithm - erm or kfold or plot', default='erm')
	args = parser.parse_args()
	data = pd.read_csv(args.dataset_path)
	data.head()
	labels = data.iloc[:, -1].values
	labels = np.where(labels == 0, -1, 1)
	instances = data.iloc[:, :-1].values
	getModeImplementation(instances, labels, args.mode)

if __name__ == '__main__':
	main()

