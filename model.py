#importing modules
from std_dataset import *
from init_para import *
from optm import *
from pred import *
from prop import *

#logistic regression model func
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    """
    Arguments:
    X_train -- training input data
    Y_train -- training output
    X_test -- test input data
    Y_test -- test output

    """

    # initialize parameters
    w, b = initialize(X_train.shape[0],X_train.shape[1])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)

    # Retrieve parameters
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    #dictionary containing information about the model
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d