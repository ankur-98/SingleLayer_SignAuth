#import modules
from prop import *

#fuction to optimize parameters
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    """    
    Arguments:
    w = weights
    b = bias
    X = input data
    Y = target output
    num_iterations = number of iterations of the optimization loop
    learning_rate = learning rate of the gradient descent

    """
    #list of all the costs computed during the optimization
    costs = []                                   
    
    for i in range(num_iterations):
        
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)     
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update parameter
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    #dictionary containing the weights w and bias b
    params = {"w": w,                           
              "b": b}
    
    #dictionary containing the gradients
    grads = {"dw": dw,                          
             "db": db}
    
    return params, grads, costs