#!/usr/bin/env python3
from scipy.io import loadmat
from scipy import signal
import numpy as np
import json

# Made by:
# Name			Studentnumber
# Niels van den Hork
# Niels van Drueten	s4496604

# Boolean to load correlation from file.
get_correlation_from_file = False

# Filename to store/load correlation matrix.
filename = "correlations.json"


# PRESENT Cipher SBox
SBox = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]


# Function f is the intermediate result,
# where i is the known non-constant data value
# and k is a small part of the key.
def f(i, k):
	return SBox[i ^ k]

# Returns the Hamming Weight of val.
def hw(val):
	return bin(val).count("1")

# Returns a Value-Prediction Matrix of size [no_inputs x no_keys]
# Input _in: Input matrix variable of size [no_inputs]
def construct_val_pred_matrix(_in, key_len):
	output = np.zeros((len(_in), 2**key_len), dtype="uint8")
	for i in range(len(_in)):
		in_elem = _in[i][0]
		for k in range(2**key_len):
			val = f(in_elem,k)
			output[i][k] = val
	return output	
	
# Returns a Power-Prediction Matrix of size [no_inputs x no_keys]
# Input _in: Value-Prediction Matrix of size [no_inputs x no_keys]
def construct_pow_pred_matrix(val_pred_matrix, key_len):
	output = np.zeros((len(_in), 2**key_len), dtype="uint8")
	for i in range(len(_in)):
		in_elem = _in[i][0]
		for k in range(2**key_len):
			val = val_pred_matrix[i][k]
			output[i][k] = hw(val)
	return output

# Uses the correlate function of the scipy io library,
# that cross-correlates two matrices.
def correlate_m(matrix1, matrix2):
	cols_matrix1 = matrix1.shape[1]
	cols_matrix2 = matrix2.shape[1]
	correlations = []
	# Debug lines
	debug = True
	if debug:
		# Do 1 correlation
		col1 = matrix1[:,[0]]
		col2 = matrix2[:,[0]]
		correlation = signal.correlate(col1, col2)
		correlations.append(correlation)
	else:
		for i in range(cols_matrix1):
			for j in range(cols_matrix2):
				col1 = matrix1[:,[i]]
				col2 = matrix2[:,[j]]
				correlation = signal.correlate(col1, col2)
				correlations.append(correlation)
				if 0 % 100 == 0:
					print("{} \t {}".format(i,j))
	return correlations


# Storing the correlation matrix in json.
def store_matrix(matrix):
	#matrix_list = matrix.tolist()
	json_string = ""
	for element in matrix:
		json_string += json.dumps(element.tolist())
	print(json_string)
	with open(filename, 'w') as outfile:
		json.dump(json_string, outfile)
	print("Correlation matrix stored in: {}".format(filename))


def load_matrix():
	return []

def sort_correlation(correlation):
	return correlation.sort()

# Opens "in.mat" file.
in_file = loadmat('in.mat')
_in = in_file['in']

# Computing value prediction matrix
val_pred_matrix = construct_val_pred_matrix(_in, 4)
print("Value prediction matrix: \n {}".format(val_pred_matrix))
# Computing power prediction matrix
pow_pred_matrix = construct_pow_pred_matrix(val_pred_matrix, 4)
print("Power prediction matrix: \n {}".format(pow_pred_matrix))

# Opens "traces.mat" file.
trace_file = loadmat('traces.mat')
_traces = trace_file['traces']
print("Traces matrix: \n {}".format(_traces))

# Computing correlation matrix
correlations = []
if get_correlation_from_file:
	print("Getting correlation matrix from file.");
	correlations = load_matrix()
else:
	print("Computing correlation matrix.");
	correlations = correlate_m(pow_pred_matrix, _traces)
		
	# Sort correlation matrix
	#sorted_correlations = sort_correlation(correlations)
	#print("sorted: {}".format(sorted_correlations))

	# Storing correlations	
	store_matrix(correlations)


# Create graphs


