import numpy as npy
from numpy import array as arr
import nnfs 

'''
DO EVERYTHING FROM SCRATCH THEN USE PYTHON

TODO: multiple linear regression from scratch
TODO: learn backpropagation
TODO: what
NOTE: can only have 1 print statement (for example purposes)

weight is a slope
biases is the y intercept 
basically multiple linear regression

'''

#inputs from 3 neurons from previous layer
inputs = [1,2,3,2.5]


#  weights and biaseses tuned knobs in an attemp to fit data

# weights for each neuron
weights = [[0.2,0.8,-0.5, 1], 
           [0.5,-0.91,0.26,-0.5],
           [0.26,-0.27,0.17,-0.87] ]

biases = [2,3,0.5]

# every neuron needs to add the sum of all (inputs * weights) + biases
# basically linear regression

#print(sum([i*w for i, w in zip(inputs,weights)]) + biases)
output = []

#for every weight and biases
for we, b in zip(weights,biases):
    output.append(sum([i*w for i, w in zip(inputs,we)]) + b)

#print(output)

#scary long print
#print([(sum([i*w for i, w in zip(inputs,we)]) + b) for we, b in zip(weights,biases)])



#dot product for a single layer
out = npy.dot(weights[0], inputs) + biases[0]

#print(out)

'''
NOTE: BATCHES
NN training is usually done on gpus 
convenient

inputs are usually passed as batches of samples
makes it easier for the NN to generalize

too big batch size is not good
usually batch size 32, 64, upt o 128
'''

#3 inputs (group of 4 each) 
# 3 outputs
inputs = [[1,2,3,2.5],
          [2,5,-1,2.0],
          [-1.5,2.7,3.3,-0.8]]

weights = [[0.2,0.8,-0.5, 1], 
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87] ]


biases = [2,3,0.5]

out = npy.dot(inputs, arr(weights).T) + biases

#print(out)


'''
INPUT DATA
usually named X

try to keep inputs and weights between -1 to 1
if too many values big, change the scale  

biases are default 0, usually changed only if the neuron is dead
unchecked leads to a dead network

'''
npy.random.seed(0)

X = [[1,2,3,2.5],
    [2,5,-1,2.0],
    [-1.5,2.7,3.3,-0.8]]


class layer_dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        #inputs (batch size) and neurons (# of neurons)
        #size of a single sample


        #makes a 2d array
        #flips the rows and columns so dont have to transpose
        self.weights = 0.1 * npy.random.randn(n_inputs, n_neurons)
        #neuron # zeros 
        self.biases = npy.zeros(n_neurons) 
    def forward(self, inputs):
        self.output = npy.dot(inputs, self.weights)+ self.biases


'''
RECTIFIED LINEAR UNIT ACTIVATION FUNCTION
'''
class ReLU:
    def forward(self, inputs:list):
        self.output = npy.maximum(0,inputs)


#input # is important, has to be like previous, but neuron # doesnt
layer1 = layer_dense(4,5) 
activation1 = ReLU()

layer2 = layer_dense(5,2)


layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

#print(layer1.output)
layer2.forward(layer1.output)
#print(layer2.output)





