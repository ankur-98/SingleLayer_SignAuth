#Packages
import numpy as np

#Function to intialize Parameters
def initialize(dim1,dim2):
    w = np.random.randn(dim1,1) * 0.01
    b = 1
    return w,b

