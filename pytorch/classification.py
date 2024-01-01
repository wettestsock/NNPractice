
import torch
from torch import nn
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.datasets import make_circles

N = '\n'

'''
 CLASSIFICATION: predicting a THING


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

X = torch.tensor(X)
y = torch.tensor(y)

print(X,N, y)


'''
GRAPHS 2 CIRCLES
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

# data to train
X_train, y_train = X[X.size()*0.8], y[y.size()*0.8]
print(X_train.size(), y_train.size())

# data to test

class circleModel(nn.Module):
    def __init__() -> None:
        super().__init__()

        # define hidden layers