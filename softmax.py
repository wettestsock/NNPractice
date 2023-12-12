import math

'''
LOSS
how wrong a model is

ReLU doesnt work for output numbers,
all negatives get cut off
no way to quantify how wrong something is

exponentiation - e^i 
normalization - i/sum of all other i 

normalization makes everything a proportion

'''
layer_outputs = [4.8,1.21,2.385]

exp_values = [math.e**i for i in layer_outputs]

#print(exp_values)
norm_values  = []


for value in exp_values:
    norm_values.append(value/sum(exp_values))


#normalized values
print(norm_values)

