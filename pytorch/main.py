import pandas as pd 
import numpy as npy 
import matplotlib.pyplot as plt 
import torch 
import torch.cuda
from torch import nn  #neural network functios


# IT WORKS
#print(t.__version__)

'''
hello and welcome to super pytorch super 

use gpu !!! a lot more!! 

python coding/NNPractice/pytorch/main.py

'''


'''
INTRO TO TENSORS
tensors: ways to represent multidimensional data

different kinds of tensors

1: scalar (0 dimensions, just 1 item)
2: vector (1 dimension, a list)
3: MATRIX (2 dimensions, a 2d list)
4: TENSOR (3 dimensions, 3d list!!)
'''

#0 dimensions
scalar = torch.tensor(7)


#print(t.tensor(7))
#print(scalar.ndim) single item has no dimensions (0)

# print(scalar.item()) #return tensor as python data type

#1 dimension
vector = torch.tensor([2,4]) # a vector

#dimensions and shape arent same
# print("vector dimensions: ", vector.ndim, "\nvector shape:", vector.shape)

#2 dimensions
MATRIX = torch.tensor([[7,8],
                       [9,10],
                       [11,11]])

#3 dimensions
TENSOR = torch.tensor([[[1,2,3],
                        [4,3,4],
                        [1,5,4],
                        [45,2,3]]])

# returns list size
# 1st dimension, 2nd, and 3rd
#print(TENSOR.shape)


'''
RANDOM TENSORS

why? 
neural networks start with tensors full of random numbers and then adjust them to better represent the data

start with random numbers -> look at data -> update random numbers -> look at data -> so on


note: size = shape
'''

#of size 3,4 

# num of elements in each dimension 
random_tensor = torch.rand(3,4)


#print(torch.rand(2,10,10,10))  can generate any dimensions

# common code way of representing images
random_image_size_tensor = torch.rand(size=(224,224,3)) #height, width, color channels
#print(random_image_size_tensor.shape, random_image_size_tensor.ndim)


#tensor of all zeros in size 3,4  
zeros = torch.zeros(3,4)
#print(zeros)

#all ones
ones = torch.ones(5,34)
#print(ones)

'''
MAKE A RANGE OF TENSORS
'''

range = torch.arange(0,10) # 0 to 10 (exclusive) so 0 <= x < 10
#print(range)

#zeros liek 
ten_zeros = torch.zeros_like(range)

#print(ten_zeros)

'''
IMPORTANT PARAMETERS

TENSORS ARE FLOAT 32 DATA TYPE BY DEFAULT
dtype = data type (from torch)

device <- where to store the tensor
device has to be the same to do manipulation

requires_grad <- calculate gradients


dtype =    # data type
device =   # where its stored (cpu gpu)
required_grad  # whether or not to track gradients
'''

float_32_tensor = torch.tensor([3.0,6.0,9.0],
                               dtype=torch.float16,
                               device='cpu',
                               requires_grad=False)

#casting
float_16_tensor = float_32_tensor.type(torch.half) 

#print(float_16_tensor)


'''
TENSOR OPERATIONS
- addition
- substraction
- multiplication (element-wise)
- division
- matrix multiplication


'''

#works same as numpy
tensor = torch.tensor([1,2,3])
tensor += 10
#print(tensor)

#matrix multiplication
# 2 ways of performing multiplication in NN and deep learning

# 1 : element-wise multiplication
# 2 : matrix multiplication <- most common  ( DOT PRODUCT)
'''
dot product: multiply each row element by its respective column element
then sum them all

2 conditions:
inner dimensions must match  
ex has to be (3,2) @ (2,3)
2=2 

(3, 2) @ (3,2) wont 
2 != 3

output: outer dimensions

(3,2) @ (2,3) = (3,3) 

ex: (1,2,3) *** (7,9,11) = 1*7 + 2*9 + 3*11 = 58

if pytorch has a method, then pytorch is faster

TRANSPOSE -

.T 
transpose method
'''

#element-wise
#print(torch.tensor([1,2,3])*torch.tensor([4,4,3]))



#matrix multiplication
#tensor_mult = torch.matmul(torch.tensor([1,2,3]), torch.tensor([4,4,3]))

# mm - alias for matmul
tensor_mult = torch.mm(torch.tensor([[1,2.0,3],[3,4,2]]), torch.tensor([[4,4.0,3],[3,4,2]]).T)
#symbol for matrix multiplication
#print(torch.tensor([1,2,3]) @ torch.tensor([4,4,3]))

# @@@@@@@@@@

#print(tensor_mult)

'''
TENSOR AGGREGATION
min/max/mean/sum/etc
'''
#print(tensor_mult.min(), tensor_mult.max(), tensor_mult.mean())


# index of the min/max 
#print(tensor_mult.argmin(), tensor_mult.argmax())


'''
TENSOR SQUEEZE AND UNSQUEEZE 

squeeze: removes single dimensions
unsuqeeze: adds single dimension to a given dimension


'''


print(tensor_mult.squeeze().shape)
print(tensor_mult.unsqueeze(0).shape)



'''
REPRODUCIBILITY
(trying to take random out of random)

start with random numbers -> tensor operations -> 
-> update random numbers to make it fit data more

'''

#have to set the seed for each block of code
torch.manual_seed(42)
manual_1 = torch.rand(3,4)


torch.manual_seed(42)
manual_2 = torch.rand(3,4)


#print(manual_1 == manual_2)


'''
runnign pytorch on GPUS
TODO: do this on my pc lmao

nvidia #1 leader on cuda cores

gpus are a lot better than cpus at numerical calculations

numpy doesnt support gpus 
tensors do
'''

# EASY SOLUTION ON PC
#print(torch.cuda.is_available())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tensor_on_cpu_for_now = torch.tensor([3,4,3])
tensor_on_cpu_for_now.to(device) # transfer to device
print(tensor_on_cpu_for_now)




'''
DATA (preparing and loading)

data can be almost anything
- spreadsheets
-images 
- videos
- audio
-DNA 
- test

1. get data into numerical representation
2. build a model to find patterns in it

jdlsfjl

sdjfkld

hi guys

sdjiflis
'''