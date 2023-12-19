import pandas as pd 
import numpy as npy 
import matplotlib.pyplot as plt 
import torch 


# IT WORKS
#print(t.__version__)

'''
hello and welcome to super pytorch super 

use gpu !!! a lot more!! 

'''


'''
INTRO TO TENSORS
tensors: ways to represent multidimensional data

different kinds of tensors

1: scalar (0 dimensions, just 1 item)
2: vector (1 dimension, a list)
3: MATRIX (2 dimensions, a 2d list)
4: TENSOR (3 dimensions, 3d list!!)
'''

#0 dimensions
scalar = torch.tensor(7)


#print(t.tensor(7))
#print(scalar.ndim) single item has no dimensions (0)

# print(scalar.item()) #return tensor as python data type

#1 dimension
vector = torch.tensor([2,4]) # a vector

#dimensions and shape arent same
# print("vector dimensions: ", vector.ndim, "\nvector shape:", vector.shape)

#2 dimensions
MATRIX = torch.tensor([[7,8],
                       [9,10],
                       [11,11]])

#3 dimensions
TENSOR = torch.tensor([[[1,2,3],
                        [4,3,4],
                        [1,5,4],
                        [45,2,3]]])

# returns list size
# 1st dimension, 2nd, and 3rd
print(TENSOR.shape)


'''
RANDOM TENSORS

why? 

'''