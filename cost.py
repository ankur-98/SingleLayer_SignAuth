import numpy as np

def J(Y,A,m,W):
    w=(A)
    x=(1-A)
    w[w == -np.inf] = 0
    x[x == -np.inf] = 0
    cost = (-1/m)*np.add(np.sum(np.multiply(Y,np.log(w))),np.sum(np.multiply((1-Y),np.log(x))))

    #L2 Regularization to decrease varraiance
    cost += (50/(2*m))*(np.dot(W.T,W))
    return cost
