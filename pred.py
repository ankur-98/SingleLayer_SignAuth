#Packages
import numpy as np

#import modules
from helper import *

#fuction to predict output
def predict(w, b, X):
    
    '''    
    Arguments:
    w = weights, a numpy array of size (num_px * num_px * 3, 1)
    b = bias, a scalar
    X = data of size (num_px * num_px * 3, number of examples)
    
    '''
    
    m = X.shape[1]                              #number of inputs
    Y_prediction = np.zeros((1,m))              #array containing all predictions
    w = w.reshape(X.shape[0], 1)
    
    #compute predicting the probabilities
    A = (sigmoid(np.add(np.dot(X.T,w),b)))
    
    for i in range(A.shape[1]):
        
        #convert probabilities to predictions
        Y_prediction = np.round(A)
    
    return Y_prediction