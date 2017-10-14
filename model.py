#importing modules
from std_dataset import *
from init_para import *
from optm import *
from pred import *
from prop import *

def accuracy(Y_prediction,Y):
    acc = 100 - np.mean(np.abs(Y_prediction - Y)) * 100
    return acc

def accept_reject_rate(true_negative, false_negative, true_positive, false_positive):

    FAR = (false_negative + 0.000000001)/(true_positive + false_negative + 0.000000001)
    FRR = (false_positive + 0.000000001)/(false_positive + true_negative + 0.000000001)
    return FAR,FRR

def confusion_matrix(Y_prediction_input,Y_input):
    count = Y_input.shape[0]
    true_negative, false_negative, true_positive, false_positive = [0, 0, 0, 0]

    print("count " + str(Y_input.shape))
    Y_input = [Y_input[i,0] for i in range(0,count)]
    Y_prediction_input = [Y_prediction_input[i,0] for i in range(0,count)]

    for i in range(count):
        if Y_input[i] == 1:
            # positive
            if Y_prediction_input[i] == Y_input[i]:
                # true
                true_positive += 1

            elif Y_prediction_input[i] != Y_input[i]:
                # false
                false_positive += 1

        elif Y_input[i] == 0:
            # negative
            if Y_prediction_input[i] == Y_input[i]:
                # true
                true_negative += 1

            elif Y_prediction_input[i] != Y_input[i]:
                # false
                true_negative += 1

    return [true_negative/count, false_negative/count, true_positive/count, false_positive/count]

#logistic regression model func
def model(X_train, Y_train, X_eval, Y_eval, num_iterations=2000, learning_rate=0.5, print_cost=False):

    """
    Arguments:
    X_train -- training input data
    Y_train -- training output
    X_eval -- eval input data
    Y_eval -- eval output

    """

    # initialize parameters
    w, b = initialize(X_train.shape[0],X_train.shape[1])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)

    # Retrieve parameters
    w = parameters["w"]
    b = parameters["b"]

    # Predict eval/train set
    Y_prediction_eval = predict(w, b, X_eval)
    Y_prediction_train = predict(w, b, X_train)

    #Accuracies
    train_accuracy,eval_accuracy = [accuracy(Y_prediction_train,Y_train),accuracy(Y_prediction_eval,Y_eval)]

    #Confusion Matrices
    train_true_negative, train_false_negative, train_true_positive, train_false_positive = confusion_matrix(Y_prediction_train,Y_train)
    eval_true_negative, eval_false_negative, eval_true_positive, eval_false_positive = confusion_matrix(Y_prediction_eval, Y_eval)

    #FAR FRR
    train_FAR, train_FRR = accept_reject_rate(train_true_negative, train_false_negative, train_true_positive, train_false_positive)
    eval_FAR, eval_FRR = accept_reject_rate(eval_true_negative, eval_false_negative, eval_true_positive, eval_false_positive)

    # Print train/eval Errors
    print("train accuracy: {} %".format(train_accuracy))
    print("evaluation accuracy: {} %".format(eval_accuracy))
    print("train FAR: {} %".format(train_FAR))
    print("train FRR: {} %".format(train_FRR))
    print("eval FAR: {} %".format(eval_FAR))
    print("eval FRR: {} %".format(eval_FRR))

    #dictionary containing information about the model
    d = {"costs": costs,
         "Y_prediction_eval": Y_prediction_eval,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "train_accuracy": train_accuracy,
         "eval_accuracy": eval_accuracy,
         "train_FAR": train_FAR,
         "train_FRR": train_FRR,
         "eval_FAR": eval_FAR,
         "eval_FRR": eval_FRR
        }

    return d