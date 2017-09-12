#Packages
import numpy as np

#Function to intialize Parameters
def initialize(dim1,dim2):
    w = np.random.random((dim1,1))
    b = 0
    return w,b

#For image inputs, w will be of shape (num_px X num_px X 3, 1)