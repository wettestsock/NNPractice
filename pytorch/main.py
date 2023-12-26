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

make data
'''

# linear regression formula - y = a+bx

weight = 0.7
bias = 0.3


#create a range of nums

start = 0
end = 1
step = 0.02

#unsqueeze - adds extra dimension, need later on
X = torch.arange(start,end,step).unsqueeze(dim=1) 


Y = weight * X + bias

#print(X[:10], Y[:10])


'''
SPLITTING DATA INTO 
TRAINING
TEST SETS

--- VERY IMPORTANT IN ML

GENERALIZATION


important to create training and test splits !!!

'''

#train 80% of the data to predict 20% of the data
train_split = int(0.8* len(X))

# 80% of the data
X_train, Y_train = X[:train_split], Y[:train_split]

# 20%
X_text, Y_text = X[train_split:], Y[train_split:]


#plt.figure(figsize=(10,7))

#plt.scatter(X_train, Y_train, c = 'b', label = 'training data')

#plt.scatter(X_text, Y_text, c='r')

#plt.show()


'''
FIRST MODEL
SIMPLE LINEAR REGRESSION
IN PYTORCH


nn.Module :

BASE CLASS FOR ALL NN MODULES
all models will subclass this class

modules can be stacked on top of each other


start with random numbers, 
adjust the weight and bias until the weight and bias fit the linear model

nn.Module: 
head class, all subclasses derived from it
modules can contain other modules (subclasses)
works like lego bricks

linReg: 
nn.Module SUBCLASS

linReg.__init__(): 
initialize model parameters to be used 
(layers, list of layers, single parameters, hard coded values, functions)

requires_grad:
gradients, basically keeps track of derivatives 
required to implement gradient descent
set by default

forward:
every subclass of nn.Module requires a forward method (forward propagation)
forward computation (relu, softmax, other activation functions)


'''


# parent class 

# linReg: nn.Module SUBCLASS
# nn.Module is the head class 

class linReg(nn.Module):
    def __init__(self):

        '''
        SUPER()
         reference to the parent class (*this but in python)
         useful for inheritance
        '''
        super().__init__()

        '''
        nn.Parameter
         special tensor that appears in Module.parameters (when used in a module)
         automatically appears as parameter in all subclasses

        
        requires_grad 
         requires a gradient

        '''

        # start with a random torch values
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))

        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # requires_grad and dtype are default, dont need to initialize 
    # hii
        
    # linear regression forward method
    def forward(self, x:torch.Tensor)->torch.Tensor: # x is the input data
        # linear regression formula (ax+b)
        return self.weights * x + self.bias 
    
'''
PROCESS

* start with random values fro weight and bias
* look at training data
* adjust random values to better represent (get closer) to the ideal values

how does it do so?

through 2 algorithms
* gradient descend
* backpropagation

usually another module from nn will define parameters for you 
for example will manually define the parameters

'''


'''
GRADIENT DESCEND

optimization algorithm for finding a local minimum of a differentiable function
multivariable calculus (billion quintillion dimensions)


differentiable function:
function where derivatives exist at all points of its domain (x values)

if cant find derivative then it its differentiable function 

gradient descent + backpropagation tweak the parameter values
'''


#lsdflj
    
'''
CHECKING CONTENTS OF RANDOM SEED

.parameters() 

cast to a list
'''

torch.manual_seed(42)

model_0 = linReg()
#print(list(model_0.parameters()))



'''
BUILDING FIRST MODEL!!

LINEAR REGRESSION
'''
class linearReg(nn.Module):

    def __init__(self):
        super().__init__()
        #real

        self.weights = nn.Parameter(torch.randn(1))
        self.biases = nn.Parameter(torch.randn(1))
    
    def forward(self, x:torch.tensor)-> torch.tensor:
        return self.weights*x + self.biases
        
# fdijfdfds