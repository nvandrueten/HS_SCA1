#!/usr/bin/env python
# coding: utf-8

from scipy.io import loadmat
from scipy import signal
import numpy as np
import json
import matplotlib.pyplot as plt
#!pip install tqdm
#from tqdm import tqdm_notebook as tqdm #if running in a notebook
from tqdm import tqdm as tqdm #if not running in a notebook
from scipy.stats.stats import pearsonr

# Made by:
# Name			Studentnumber
# Niels van den Hork 	s4572602
# Niels van Drueten 	s4496604


# PRESENT Cipher SBox
SBox = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]


# Function f is the intermediate result,
# where _in is the known non-constant data value
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
def construct_pow_pred_matrix(_in, val_pred_matrix, key_len):
	output = np.zeros((len(_in), 2**key_len), dtype="uint8")
	for i in range(len(_in)):
		in_elem = _in[i][0]
		for k in range(2**key_len):
			val = val_pred_matrix[i][k]
			output[i][k] = hw(val)
	return output


# Uses the person correlate function
# that cross-correlates two matrices.
def correlate_m(matrix1, matrix2):
    print("Computing correlation matrix of 2 matrices with size {} and {}".format(matrix1.shape,matrix2.shape))
   
    cols_matrix1 = matrix1.shape[1]
    cols_matrix2 = matrix2.shape[1]
    
    vmax = -1e10 #max correlation value
    cmax = (-1,-1) # coordinate indices with max correlation
    
    
    result = np.zeros((cols_matrix1,cols_matrix2)) # sneller, want geen appends
    for i in tqdm(range(cols_matrix1)):
        for j in range(cols_matrix2):
            #result[i][j] = signal.correlate(matrix1[:,[i]], matrix2[:,[j]])[0] # dit is correlation als in convolution, niet pearson correlation
            #print(result[i][j])
            result[i][j] = pearsonr(matrix1[:,[i]], matrix2[:,[j]])[1][0] #[0][0] returns p value
            #result[i][j] = numpy.corrcoef(matrix1[:,[i]], matrix2[:,[j]])[0, 1]
            
    return result

def sort_correlation(correlation):
	return list(sorted(correlation,key= lambda x: x[0][0]))


def main():
	# Opens "in.mat" file.
	in_file = loadmat('in.mat')
	_in = in_file['in'] #contains 14900 4bit inputs

	print("Input: \n {} \nSize: {}".format(_in, _in.shape))


	# Computing value prediction matrix
	val_pred_matrix = construct_val_pred_matrix(_in, 4)
	print("Value prediction matrix: \n {} \nSize: {}".format(val_pred_matrix, val_pred_matrix.shape))


	# Computing power prediction matrix
	pow_pred_matrix = construct_pow_pred_matrix(_in, val_pred_matrix, 4)
	print("Power prediction matrix: \n {} \nSize: {}".format(pow_pred_matrix, pow_pred_matrix.shape))
	#plt.figure(1)
	#plt.plot(pow_pred_matrix[:,0])


	# Opens "traces.mat" file.
	trace_file = loadmat('traces.mat')
	_traces = trace_file['traces']
	print("Traces matrix: \n {} \nSize: {}".format(_traces, _traces.shape))
	#plt.figure(2)
	#plt.plot(_traces[0])


	# Computing correlation matrix nr_of_traces x nr_of_keys
	correlation_matrix = correlate_m(pow_pred_matrix, _traces)
	print("Correlation matrix: \n {} \nSize: {}".format(correlation_matrix, correlation_matrix.shape))
	print(correlation_matrix[0])
	# Sort correlation matrix
	#sorted_correlations = sort_correlation(correlation_matrix)
	#print("sorted: {} \nSize: {}".format(sorted_correlations, sorted_correlations.shape))

	# Plotting correlation matrix
	for row in correlation_matrix:
		plt.plot(row)
	plt.figure(1)
	plt.show()

	# Plotting sorted correlation matrix
	for row in correlation_matrix:
		row = row.sort()
	for row in correlation_matrix:
		plt.plot(row)
	plt.figure(2)
	plt.show()

	# Best key candidate 
	print(correlation_matrix[0][:-2])

	plt.hist(correlation_matrix[0])
	plt.figure(3)
	plt.show()

if __name__== "__main__":
	main()

