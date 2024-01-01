
import torch
import torch.nn 
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

X = torch.tensor(X[:,0])
y = torch.tensor(y)

print(X,N, y)

plt.scatter(X[:,0], y)
plt.show()