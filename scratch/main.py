import numpy as npy
import matplotlib as plt 

print('hi')
#build neural network in c++ when possible lmao 


#INTRODUCTION
inputs = [1,2,3] # every unique input has unique weight associated with it
#note: inputs are either actual input layer data or output from the previous hidden layer
# cant be adjusted 
# weights and bias have to be calculated using backpropagation and calculating gradient

weights = [0.2,0.8,-0.5] 
bias = 2  #every neuron has its own bias

#step 1: inputs * weights + bias
neuron1OfNextLayer = inputs[0]*weights[0]+inputs[1]*weights[1]+inputs[2]*weights[2] + bias  
# every input * every weight for each input to that specific neuron + bias of the neuron
print(neuron1OfNextLayer)

inputs = [1,2,3,2.5] # cant change inputs, 
weights1=[0.2,0.8,-0.5,1]
weights2=[0.5,-0.91,0.26,-0.5]
weights3=[-0.26,-0.27,0.17,0.87]

bias1,bias2,bias3=2,3,0.5

layerOutput= [
    inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
    inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
    inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3
    ]

print('layer output is:',layerOutput)
