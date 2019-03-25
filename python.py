#!/usr/bin/env python3
from scipy.io import loadmat


# PRESENT Cipher SBox
SBox = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]


# Function f is the intermediate result,
# where _in is the known non-constant data value
# and k is a small part of the key.
def f(_in, k):
	return SBox[_in ^ k]


x = loadmat('in.mat')
print(x['in'][4][0])

print(f(1,0))
