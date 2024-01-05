
import torch
from torch import nn
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.datasets import make_circles

N = '\n'

'''
 CLASSIFICATION: predicting a THING
sdjfls
TEST
TEST 2:
 BINARY CLASSIFICATION:
 yes, no 
 2 choices

 MULTICLASS CLASSIFICATION:
 has multiple images and assigns 1 label to EACH


 MULTILABEL CLASSIFICATION:
 assigning multiple labels/tags to a single image


 classification inputs and outputs:


 BACH SIZE:
 32 inputs at a time and they output the probabilites, compared to train outputs
 
 32 is very common batch size because it's efficient
'''

n_samples = 1000

# make_circles returns x and y 
X, y = make_circles(n_samples, # 1000 samples
                    noise=0.03, #randomness
                    random_state=42) # random seed

X = torch.tensor(X,dtype=float)
y = torch.tensor(y, dtype=float)

#print(X,N, y)


'''
GRAPHS 2 CIRCLES

inputs are of list length 2 (very small)
can be represented by x and y 

could be working with millions of length with millions of dimensions


WANT TO PREDICT IF GIVEN X AND Y COORDINATES ARE THE OUTER OR INNER CIRCLE

note: data worked in is often called as a toy dataset
TOY DATASET: data small enough to experiment but big enough to stabilize
'''
# c - classify (separates data)
#plt.scatter(X[:,0], X[:,1], c=y)
#plt.show()


'''
FINALLY MAKING A MODEL

1. subclasses nn.Module
2. 2 nn.Linear() layers (hidden layers) (can handle the shapes of data)
3. define forward method
4. instantiate instance of model class and send to target device


'''

# 80 20 split
split_80_20 = int(len(X)*0.8)


# data to train
X_train, y_train = X[:split_80_20], y[:split_80_20]


# data to test
X_test, y_test = X[split_80_20:], y[split_80_20:]

#check the size
#print(X_train.size(), y_train.size(), X_test.size(), y_test.size())

class circleModel(nn.Module):
    def __init__(self):
        super().__init__()


        '''
        nn.Linear:
            linear layer
        
            random when initialized
            linear transofrmation to the incoming data
            applies linear y=mx+b transformation to the vector

        '''
        # input layer 
        # 2 features (X has 2 features)
        # out features is 5 (hidden layer) 


        self.layer_1 = nn.Linear(in_features=2, # per neuron 
                                 out_features=8) # hidden layer output, are usually a multiple of 8
        
        # shape features have to match the previous out
        self.layer_2 = nn.Linear(in_features=8, # hidden layer input
                                 out_features=1) # 1 output (binary classification), the output layer
        # define hidden layers

        '''
        self.two_linear_layers = nn.Sequential(
           nn.Linear(in_features=2, out_features=8),
           nn.Linear(in_features=8, out_features=1) 
        )
        '''

    # forward pass
    def forward(self, x):
        # output of layer 1 goes into input of layer 2 and returns that
        return self.layer_2(self.layer_1(x))
    

# instantiate model class and send to target device
    
model_0 = circleModel().to(device='cpu') # model to cpu


'''
nn.Sequential:
    has the input layers and sequentially iterates over them
    same as the coded class 

    however, sublasses are for more complicated models
'''

'''
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=8),
    nn.Linear(in_features=8, out_features=1)
).to('cpu')
'''


'''
layer 1: 
    2 input features, 8 output features, 
    and bias for wach output feature

    and bias

layer 2:
    8 input features, 1 output feature 
    1 bias for the 1 output
'''

print(model_0.state_dict())
#fdijgjflgdf


'''
LOSS FUNCTION AND OPTIMIZER
'''

'''
loss: MAE (mean absolute error)
doesnt work for classification

for regression: MAE or MSE (mean squared error)

for classification: binary cross-entropy or categorical cross-entropy
for classification: SGD and adam optimizers

for binary: sigmoid is better for output layer
for classification: softmax is better

nn.BCELoss

more stable and faster than creating them separately
nn.BCEWithLogitsLoss  (combines sigmoid layer and BCELoss)


for multi class classification: cross entropy loss
'''

'''
LOGITS LAYER:

layer that feeds into softmax activation function (output layer) - outputs are probabilities

for optimizers: SGD and adam, however, pytorch has more
^ problem-specific

LOGITS:
if probability over .5, make it 1
if probability under .5, make it 0

'''

#calculate accuracy, out of 100 samples what percent does model get right?
def accuracy_fn(y_true, y_pred):

    # correct # 
    correct = torch.eq(y_true, y_pred).sum().item() #item: single value

    return (correct/len(y_pred)) * 100


'''
model outputs will be raw LOGITS
convert logits into prediction probabilities into a activation function
(sigmoid for cross entropy and softmax for multiclass)

layers and activation functions are separate

'''


'''
BUILDING MODEL
'''


'''
takes logits (0 and 1) as inputs!!

BCELoss takes prediction probabilities
'''
loss_fn = torch.nn.BCEWithLogitsLoss()

'''
SDG !!
'''

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

torch.manual_seed(42)

epochs = 100

for epoch in range(epochs):
    model_0.train() # TRAINING MODE

    #forward pass
    y_logits = model_0(X_train).squeeze()

    # turns logits into pred probs into pred labels (binary)
    y_pred = torch.round(torch.sigmoid(y_logits))

    # calculate loss/accuracy percentage
    '''
    without logits 

    loss = loss_fn(torch.sigmoid(y_logits),
                   y_train)
    '''
                   
    loss = loss_fn(y_logits,
                   y_train)
    
    # custom accuracy function, just percent
    acc = accuracy_fn(y_train, y_pred)

    optimizer.zero_grad()

    # calculate the gradients 
    loss.backward()

    # tweak the parameters according to gradients
    optimizer.step()


model_0.eval()
with torch.inference_mode():
    test_logits = model_0(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))


