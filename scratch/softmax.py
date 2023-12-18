import math
import numpy as npy 

'''

SOFTMAX ACTIVATION  FUNCTION 
LOSS
how wrong a model is
WHATS THE PROBABILITY OF SOMETHING BEING CORRECT

input -> exponentiate -> normalize -> output

ReLU doesnt work for output numbers,
all negatives get cut off
no way to quantify how wrong something is

exponentiation - e^i 
normalization - i/sum of all other i 

normalization makes everything a proportion

'''
layer_outputs = [4.8,1.21,2.385]

#exp_values = [math.e**i for i in layer_outputs]
exp_values = npy.exp(layer_outputs) #apply to each value

#NOTE: print(exp_values)
norm_values  = exp_values/npy.sum(exp_values)


#for value in exp_values:
#    norm_values.append(value/sum(exp_values))


#normalized values
#NOTE: print(norm_values)


layer_outputs = [[4.8,1.21,2.385],
                 [8.9,-1.81,0.2],
                 [1.41,1.051,0.026]]


#exponential values

#to prevent exploding values, move all to 

exp_values = npy.exp(layer_outputs)
#NOTE: print(exp_values)

#NOTE: print(npy.sum(layer_outputs, axis=1, keepdims=True)) #axis: direction to add
#axis default none (adds all)
# axis = 0 vertically 1 horizontally

#keepdims: keeps the dimensions 
#default none

norm_values = exp_values / npy.sum(exp_values, axis = 1, keepdims=True)

#NOTE: print(norm_values)

'''
SOFTMAX ACTIVATION FUNCTION EASY TO OVERFLOW,
EXPONENTIAL


overflow prevention:

do this before exponentiation
solution: substract biggest number from each value
biggest always becomes 0 

'''

