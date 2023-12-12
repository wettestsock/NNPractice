

'''
DO EVERYTHING FROM SCRATCH THEN USE PYTHON

TODO: multiple linear regression from scratch
TODO: learn backpropagation
TODO: what


'''

#inputs from 3 neurons from previous layer
inputs = [1,2,3,3.5]


#  weights and biases tuned knobs in an attemp to fit data

# weights for each neuron
weights = [[0.2,0.8,-0.5, 1], 
           [0.5,-0.91,0.26,-0.5],
           [0.26,-0.27,0.17,-0.87] ]

bias = [2,3,0.5]

# every neuron needs to add the sum of all (inputs * weights) + bias
# basically linear regression

#print(sum([i*w for i, w in zip(inputs,weights)]) + bias)
output = []

#for every weight and bias
for we, b in zip(weights,bias):
    output.append(sum([i*w for i, w in zip(inputs,we)]) + b)

print(output)

#scary long print
#print([(sum([i*w for i, w in zip(inputs,we)]) + b) for we, b in zip(weights,bias)])



