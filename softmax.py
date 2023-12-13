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

#print(exp_values)
norm_values  = exp_values/npy.sum(exp_values)


#for value in exp_values:
#    norm_values.append(value/sum(exp_values))


#normalized values
#print(norm_values)


layer_outputs = [[4.8,1.21,2.385],
                 [8.9,-1.81,0.2],
                 [1.41,1.051,0.026]]

exp_values = npy.exp(layer_outputs)
#print(exp_values)

print(npy.sum(layer_outputs, axis=1)) #axis: direction to add
#axis default none (adds all)
# axis = 0 vertically 1 horizontally
