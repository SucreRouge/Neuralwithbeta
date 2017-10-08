# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:02:59 2017

@author: Aditya
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import numpy as np
from minepy import MINE
import sys
from datetime import datetime
import re
import pandas as pd

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    #hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    hidden_layer = [{'weights': [0.5 for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    #hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network. append(hidden_layer)
    #output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    output_layer = [{'weights':[0.5 for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network. append(output_layer)
    return network

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

def transfer(activation):
    return 1.0/ ( 1 + exp (-activation))

def transfer_derivative(output):
    return output * (1.0 - output)

def squasher(beta):
    return 1.0/ ( 1 + exp (-beta))

def squasher_derivative(output):
    return squasher(output) * (1.0 - squasher(output))

def calc_delta_betas(network, inp_vector, betas): # this returns delta beta without taking learning rate into account
    delta_betas = list()
    for i in range(len(inp_vector)-1):
        inputj = inp_vector[i]
        summ = 0.0
        for neuron in network[0]:
            summ += neuron['weights'][i] * neuron['delta']
        delta_betas.append( (-1.0) * inputj * (summ) * squasher_derivative(betas[i]))
    return delta_betas


def forward_propogate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])

        inputs = new_inputs
    summ = 0
    #for neuron in network[len(network) - 1]:
     #   summ += neuron['output']
    #print summ
    return inputs


def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])

		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])



def update_weights(network, row, l_rate):

    for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']



"""def train_network(network, train, l_rate, n_epoch, n_outputs):
      betas = [ 0.008  for i in range(len(train[0]) - 1) ]
      for epoch in range(n_epoch):
          sum_error = 0
          errors = list()
          for t_row in train:
                #print t_row

                row = [squasher(betas[i]) * t_row[i] for i in range(len(t_row) - 1)]
                row.append(t_row[-1])
                outputs = forward_propogate(network, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                errors. append(sum_error)
                backward_propagate_error(network, expected)
                update_weights(network, row, l_rate)
                delta_beta = calc_delta_betas(network, t_row, betas)
                betas = [ betas[i] +  0.01  * delta_beta[i] for i in range(len(betas))]

                print  ('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
                print "\n"
      #print ([neuron['weights'] for neuron in network[0]])
      all_betas.append(betas)
"""

def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            for g in row:
                    print " %1.5f" % (g) ,
            print
            outputs = forward_propogate(network, row)
            for g in outputs:
                    print " %1.5f" % (g) ,
            print

            expected = [0 for i in range(n_outputs)]
            #print(row[-1])
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
            print 'weights :'
            for neuron in network[1]:
                for wts in neuron['weights']:
                    print  wts,

                print
            print expected
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
            print ("\n")
    #print ([neuron['weights'] for neuron in network[0]])

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

#Convert string column to float

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())


# Convert string column to integer

def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for i  in range(len(folds)):
        fold = folds[i]
        train_set = list(folds)
        train_set. remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)

            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, fold , *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores

def back_propagation(train, test, fold, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()

    for i in range(len(test)):
        row = test[i]
        prediction = predict(network, row)
        predictions. append(prediction)
    return(predictions)

def predict(network, row):
    outputs = forward_propogate(network, row)
    return outputs.index(max(outputs))
#network = initialize_network(2, 1, 2)

def cor(g):
    corr = list()
    for i in range(len(g)):
        base = np.array(list(g[i]))
        l = list()
        for j in range(len(g)):
            comp = np. array(list(g[j]))
            m = MINE( )
            m.compute_score(base, comp)
            l.append(m.mic())
        corr. append(l)
    return corr


# load and prepare data

"""
sys.stdout = open('log8.txt', 'w')
filename = 'outpu23t.csv'
seed(1)
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
dataset_new = list()
for row in dataset:
    n_row = []
    n_row = row[0 : len(row) -1 ]
    n_row. append(random())
    n_row.append(random())
    n_row. append(row[-1])
    dataset_new. append(n_row)
"""

dataframe = pd.read_csv('')
dataset = dataframe.values
dataset = dataset_new


#dataset = dataset_new
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
data = list()
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 100
n_hidden = 7
all_betas = list()
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print "All Betas- "

for betas in all_betas:
   for g in betas:
       print "\t %1.5f" % (g) ,
   print

print("\n")
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (float(sum(scores) ) / float(len(scores))))
