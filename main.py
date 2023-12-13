import numpy as npy
from numpy import array as arr
import nnfs
import nnfs.datasets.spiral as spiral

nnfs.init()

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

#NOTE: print(sum([i*w for i, w in zip(inputs,weights)]) + biases)
output = []

#for every weight and biases
for we, b in zip(weights,biases):
    output.append(sum([i*w for i, w in zip(inputs,we)]) + b)

#NOTE: print(output)

#scary long print
#NOTE: print([(sum([i*w for i, w in zip(inputs,we)]) + b) for we, b in zip(weights,biases)])



#dot product for a single layer
out = npy.dot(arr(weights[0]), arr(inputs)) + biases[0]

#NOTE: print(out)

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

out = npy.dot(arr(inputs), arr(weights).T) + biases

#NOTE: print(out)


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




'''
CLASSES




'''

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


# RELU 
class ReLU:
    def forward(self, inputs:list):
        self.output = npy.maximum(0,inputs)


class softmax:
    def forward(self, inputs):
        #exponentiate
        exp_values = npy.exp(inputs- npy.max(inputs, axis = 1, keepdims=True))
        #keeps dimensions of the inputs 
        #max to prevent overflow

        #normalize
        probabilities = exp_values/npy.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

#input # is important, has to be like previous, but neuron # doesnt
layer1 = layer_dense(4,5) 
activation1 = ReLU()

layer2 = layer_dense(5,2)


#course materials
X, y = spiral.create_data(samples = 100, classes=3)

#input 2 (x and y coordinates)
dense1 = layer_dense(2,3)
activation1 = ReLU()


dense2 = layer_dense(3, 3)
activation2 = softmax()


dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)


#when model is initialized, the probabilities are random
print(activation2.output[:5])





