
import numpy as npy
import matplotlib as plt 

print('hi')
#build neural network in c++ when possible lmao 


#INTRODUCTION
inputs = [1,2,3] # every unique input has unique weight associated with it
#note: inputs are either actual input layer data or output from the previous hidden layer
# cant be adjusted 
# weights and bias have to be calculated using backpropagation and calculating gradient
# can have neural network with just bias or weights , but most of the time all are there


#   bias is done to offset the weight to make it positive

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

# more dynamic way to do this:
weights = [[0.2,0.8,-0.5,1], 
           [0.5,-0.91,0.26,-0.5], 
           [-0.26,-0.27,0.17,0.87]]
biases = [2,3,0.5]

print('zip list is:',list(zip(weights,biases)))

layer_outputs = []
for nWeights, nBias in zip(weights, biases): # makes a list of lists each of which has elements from both lists
    nOutput=0
    for nInput, weight in zip(inputs, nWeights):
        nOutput+=nInput*weight
    nOutput+=nBias
    layer_outputs.append(nOutput)

print(layer_outputs)


outputs= npy.dot(inputs,weights[0])+bias  #dot product of 2 vectors 
                    #sum of all inputs*weights 

print(outputs)
weights = [[0.2,0.8,-0.5,1], 
           [0.5,-0.91,0.26,-0.5], 
           [-0.26,-0.27,0.17,0.87]]
biases = [2,3,0.5]


outputs= npy.dot(weights,inputs)+biases #weights is not a matrix
            #matrix always goes before the vector
            #basically does weights C1 * inputs then weights C2 * inputs so on and so forth
            
            #first element passed will determine how return how return value indexed 
print(outputs)

#convert single sample of inputs to a batch


#each 1d array is within a given neuron
#cant fit all samples at once because overfitting
#insample data would be really good but outsample data would suck 
inputs = npy.array([[1,2,3,2.5],
          [2,5,-1,2.0],
          [-1.5,2.7,3.3,-0.8]])


weights = npy.array([[0.2,0.8,-0.5,1], 
           [0.5,-0.91,0.26,-0.5], 
           [-0.26,-0.27,0.17,0.87]])

biases = npy.array([2,3,0.5]) #biases # = # of outputs (bias per each output neuron)


#weights and biases of the 2nd layer
weights2 = npy.array([[0.1,-0.14,0.5],  
           [-0.5,0.12,-0.33], 
           [-0.44,0.73,-0.13]])

biases2 = npy.array([-1,2,-0.5])


layer1_output =npy.dot(inputs, weights.T)+biases  #T is transpose  # bias is added to each row 
#outputs becomes matrix of rows row size = weights row size & column size = transposed inputs size
#4 input neurons, 3 output neurons


layer2_output=npy.dot(layer1_output, weights2.T)+biases2 # biases is a vector
#3 inputs 3 outputs

print('scalar values (inputs for layer 2) for layer 1 :\n',npy.array(layer1_output).round(2), '\n')
print('scalar values (outputs)for layer 2:\n',npy.array(layer2_output).round(2), '\n')

#---------------------


#MOREEEEEEEEEEEEEEEEEEEEEEEEEEE


#changing size n number of layers

#WEIGHTS -----
#weights are usually -1 to 1 
#small values preferred, if each value is bigger than 1 then it explodes
#if bigger than 1, try to scale them down so that most are -1 to 1

#---

#biases tend to be 0 initially but change if input neuron is 0
# ^^^^  then it multiplied by anything will still be 0, thus leading to a dead neuron and eventually a dead network

#---

# "hidden" layer -> layers not in charge of adjusting, NN tweaks that 

#---

#sample data
# X is the standard in ML for inputs
#also X_train and X_test
X = npy.array([[1,2,3,2.5],
          [2,5,-1,2.0],
          [-1.5,2.7,3.3,-0.8]]) 



npy.random.seed(0)



    


#g