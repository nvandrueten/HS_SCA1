#!/usr/bin/env python3
from scipy.io import loadmat
import numpy as np

# PRESENT Cipher SBox
SBox = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]


# Function f is the intermediate result,
# where _in is the known non-constant data value
# and k is a small part of the key.
def f(_in, k):
	return SBox[_in ^ k]


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
	
def print_r(ndarray):
	for i in range(len(ndarray)):
		print(ndarray[i])

matlab_file = loadmat('in.mat')
_in = matlab_file['in']


val_pred_matrix = construct_val_pred_matrix(_in, 4)

print(val_pred_matrix)
 
