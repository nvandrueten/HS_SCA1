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
#from numpy import correlate as corr #not pearson 

# Made by:
# Name			Studentnumber
# Niels van den Hork 	s4572602
# Niels van Drueten	s4496604



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
    print(matrix1.shape,matrix2.shape)
   
    cols_matrix1 = matrix1.shape[1]
    cols_matrix2 = matrix2.shape[1]
    
    result = np.zeros((cols_matrix1,cols_matrix2))
    
    for i in tqdm(range(cols_matrix1)):
        for j in range(cols_matrix2):
            #result[i][j] = pearsonr(matrix1[:,i], matrix2[:,j])[0]
            result[i][j] = np.corrcoef(matrix1[:,i], matrix2[:,j])[0][1]
    return result

# Step 1:
# Opens "in.mat" file.
in_file = loadmat('in.mat')
_in = in_file['in'] #contains 14900 4bit inputs
print("Input: Size {}\n {}".format(_in.shape, _in))

# Step 2:
# Computing value prediction matrix
val_pred_matrix = construct_val_pred_matrix(_in, 4)
print("Value prediction matrix: Size {}\n {}".format(val_pred_matrix.shape, val_pred_matrix))

# Step 3:
# Computing power prediction matrix
pow_pred_matrix = construct_pow_pred_matrix(val_pred_matrix, 4)
print("Power prediction matrix: Size {}\n {}".format(pow_pred_matrix.shape, pow_pred_matrix))

# Step 4:
# Opens "traces.mat" file.
trace_file = loadmat('traces.mat')
_traces = trace_file['traces']
print("Traces matrix: Size {}\n {}".format(_traces.shape, _traces))

# Step 5:
# Computes correlation between 
# power prediction matrix and traces matrix.
result = correlate_m(pow_pred_matrix, _traces)
print("Correlation matrix: Size {}\n {}".format(result.shape, result))


# Step 6:
# Plotting absolute correlation of key candidates.
plt.plot([sum(list(map(abs,row))) for row in result])
plt.suptitle('Key Candidates based on absolute correlation')
plt.xlabel('key')
plt.ylabel('summed absolute correlation')
plt.show()

absresult = np.array([list(map(abs,row)) for row in result])
maxidx = np.argmax(absresult,axis=1)

maxconf = np.array([(row[0],midx,row[1][midx]) for row,midx in zip(enumerate(absresult),maxidx)])
smaxconf = np.array(sorted(maxconf,key = lambda x : -x[2]) )

#[print(e) for e in smaxconf]
    
plt.bar(range(16),maxconf[:,2] )
plt.suptitle('Key Candidates based on absolute correlation')
plt.xlabel('key')
plt.ylabel('peak absolute correlation')
plt.show()

# Step 7:
for i,row in enumerate(result):
    if i == 6:
        continue
    plt.plot(list(map(abs,row)),color='gray')
    
plt.plot(list(map(abs,result[6])),color='blue')
plt.suptitle('Absolute correlation of every key candidate (blue = 6)')
plt.xlabel('time samples')
plt.ylabel('absolute correlation')
plt.show()


# Step 8: 
keyranking = []
for amount in [500,1000,2000,4000,8000,12000]:
    result = correlate_m(pow_pred_matrix, _traces[:,:amount])
    
    absresult = np.array([list(map(abs,row)) for row in result])
    maxidx = np.argmax(absresult,axis=1)

    maxconf = np.array([(row[0],midx,row[1][midx]) for row,midx in zip(enumerate(absresult),maxidx)])
    smaxconf = np.array(sorted(maxconf,key = lambda x : -x[2]) )
    #[print(e) for e in smaxconf]
    
    keyrank = np.array([e[0] for e in smaxconf])
    keyidx = np.where(keyrank == 6)[0][0]
    print(keyidx)
    keyranking.append(keyidx)


plt.plot(np.array([500,1000,2000,4000,8000,12000]),keyranking)  
plt.suptitle('Ranking of key=6') 
plt.xlabel('amount of timesamples')
plt.ylabel('ranking (lower is better)')
plt.show()
