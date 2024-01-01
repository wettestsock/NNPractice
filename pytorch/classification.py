
import torch
import torch.nn 
from pathlib import Path



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

