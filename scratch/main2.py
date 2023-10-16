import numpy as npy
import nnfs #forces the data type, just to replicate datatype
from nnfs.datasets import spiral_data
from numpy import array
# RECTIFIED LINEAR ACTIVATION FUNCTION
#best to use at least 2 hidden layers
nnfs.init() 

X = npy.array([[1,2,3,2.5],
          [2,5,-1,2.0],
          [-1.5,2.7,3.3,-0.8]])

X, y = spiral_data(100,3) #100 features of 3 classes
print(X,y)

inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = []

for i in inputs:
    # easier in python:
    output.append(max(0,i))
    #max of either 0 or i (if i is smaller than 0 then appends 0)
    '''
    if i>0:
        output.append(i)
    elif i<=0:
        output.append(0)
    '''

print(output)


class layer_dense:
    def __init__(self, n_inputs, n_neurons): #inputs: size fo the input data (# of features per sample), neurons: neuron output
        self.weights = 0.1*npy.random.randn(n_inputs,n_neurons)
        self.biases = npy.zeros((1,n_neurons))
    def forward(self,inputs): #input: inputs for each output neuron
        self.output = npy.dot(inputs,self.weights) + self.biases
        
class activation_ReLU:
    def forward(self,inputs):
        self.output = npy.maximum(0,inputs)

layer1 = layer_dense(2,5)
activation1= activation_ReLU()

layer1.forward(X) 
activation1.forward(layer1.output)
print(activation1.output) # if theres zeros: introduce bias


#TODO: pip install nnfs on pc 