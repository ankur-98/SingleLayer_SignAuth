import numpy as np

def J(Y,A,m):
    w=np.log(A)
    x=np.log(1-A)
    w[w == -np.inf] = 0
    x[x == -np.inf] = 0
    cost = (-1/m)*np.add(np.sum(np.multiply(Y,np.log(w))),np.sum(np.multiply((1-Y),np.log(x))))
    return cost
