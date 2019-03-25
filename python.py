#!/usr/bin/env python3
from scipy.io import loadmat
from scipy import signal
import numpy as np

# Made by:
# Name			Studentnumber
# Niels van den Hork
# Niels van Drueten	s4496604


# PRESENT Cipher SBox
SBox = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]


# Function f is the intermediate result,
# where _in is the known non-constant data value
# and k is a small part of the key.
def f(_in, k):
	return SBox[_in ^ k]

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
	for i in range(cols_matrix1):
		for j in range(cols_matrix2):
			col1 = matrix1[:,[i]]
			col2 = matrix2[:,[j]]
			#correlation = signal.correlate(col1, col2)
			#print(correlation)

# Opens "in.mat" file.
in_file = loadmat('in.mat')
_in = in_file['in']

val_pred_matrix = construct_val_pred_matrix(_in, 4)
print("Value prediction matrix: \n {}".format(val_pred_matrix))
pow_pred_matrix = construct_pow_pred_matrix(val_pred_matrix, 4)
print("Power prediction matrix: \n {}".format(pow_pred_matrix))

# Opens "traces.mat" file.
trace_file = loadmat('traces.mat')
_traces = trace_file['traces']

correlation = correlate_m(pow_pred_matrix, _traces)
print(correlation)
