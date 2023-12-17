import numpy as npy
from numpy import array as arr
import nnfs
import nnfs.datasets.spiral as spiral
import math

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

python coding/NNPractice/main.py

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


class loss:
    def calculate(self, output, target):
        #loss for each sample 
        sample_losses = self.forward(output,target) 
        
        #average loss
        data_loss = npy.mean(sample_losses)

        return data_loss

#derived class
class loss_categorical(loss):
    def forward(self, y_pred, y_true):
        #sample length
        samples = len(y_pred) 

        #clip all values close to 0 (1e-7 to 1-1e-7)
        y_pred_clipped = npy.clip(y_pred, 1e-7, 1-1e-7)

        '''
        TARGETS ARE EITHER PASSED AS:
        both are used

        scalar values:
        [1,0]

        or one-hot encoded:
        [[0,1],[1,0]]

        ex 2: 
        scalar: [0,2,1]
        one-hot: [[1,0,0],[0,0,1],[0,1,0]]

        '''
        if len(y_true.shape) == 1: #true scalar values form
            correct_confidences = y_pred_clipped[range(samples), y_true] #1st, 2nd dimensions

            '''
            range(samples): reference all the samples within the batch (samples is length of predicted)
            y_true: grabs the values in the indexes of the scalar values

            '''
        elif len(y_true.shape) == 2: #one hot encoded values
            correct_confidences = npy.sum(y_pred_clipped*y_true, axis =1)


            '''
            both are 2d arrays
            multiplying is easier

            axis=1: horizontal
            '''
        
        #loss
        neg_log_prob = -npy.log(correct_confidences)
        return neg_log_prob #comes back to sample losses

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
#NOTE: print(activation2.output[:5])


loss_fn = loss_categorical()
loss = loss_fn.calculate(activation2.output, y)

print('Loss:', loss)



#rewritten functions





'''
LOGS:

e**x = b , find x from b

log(b) = x 

e raised to what = b 

'''


'''
CATEGORICAL CROSS-ENTROPY 

loss 
is this loss?

one-hot encoding: 
classes: amount of elements, all 0 
labels: index of a 1 

loss = -log(observed value * expected value)
'''
#ex
softmax_output = [0.7,0.1,0.2] #output for 1 node
target_output = [1,0,0] #target output for 

# 0 is the label
#loss = -(math.log(softmax_output[0]*target_output[0]))

#label: correct index
def loss(obs:list, label:int = 0)->float:
    return -(math.log(obs[label]))

#NOTE: print(loss(softmax_output))

'''
LOSS IMPLEMENTATION
'''

softmax_output = arr([[0.7,0.1,0.2],
                      [0.1,0.5,0.4],
                      [0.02,0.9,0.08]])

class_targets = [0,1,1]

#this has a problem
loss = -npy.log(
    softmax_output[range((len(softmax_output))), class_targets]
    ) #1st dimension, 2nd

average_loss = npy.mean(loss)


#NOTE: print(average_loss)

#cant log 0, it's infinite
#solution: clip all zeros to insignificant number

'''
OPTIMIZATION AND DERIVATIVES

how to adjust weights and biases to reduce loss?
cant be random, especially for non linear datasets

certain weights and biases weight more than others
calculus helps


'''


