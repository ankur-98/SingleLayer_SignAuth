#Packages
import numpy as np

#import modules
from helper import *
from cost import *

#fuction for forward and backward propagation
def propagate(w, b, X, Y):
    
    """
    Arguments:
    w = weights
    b = bias
    X = input
    Y = target output
    
    """
    
    m = X.shape[1]

    # FORWARD PROPAGATION
    A = (sigmoid(np.add(np.dot(X.T,w),b)))         # compute activation
    cost = J(Y,A,m)     # compute cost
    
    # BACKWARD PROPAGATION 
    dz = np.add(A,-Y)
    dw = (1/m)*(np.dot(X,dz))   #dw = gradient of the loss with respect to w, thus same shape as w
    db = (1/m)*(np.sum(dz))       #db = gradient of the loss with respect to b, thus same shape as b
    cost = np.squeeze(cost)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost