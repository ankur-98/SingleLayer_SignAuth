#Packages
import numpy as np

#function to get size of datasets
def get_m(train,eval):
    m_train = train.shape[0]     #Number of training examples
    m_eval = eval.shape[0]       #Number of evaling examples
    num_px = train.shape[1]      #Height/Width of each image
    #Each image is of size: (300, 300, 3)
    return m_train,m_eval,num_px

#function to return flattened image dataset
def get_flatten(input):
    input_flatten = input.reshape(input.shape[0],-1).T
    return input_flatten

#function to standardize dataset
def standardize(input_f):
    input = input_f/255
    return input
                                            
train_set_x_orig, eval_set_x_orig, train_set_y, eval_set_y = np.load('data_set.npy')
m_train,m_eval,num_px = get_m(train_set_x_orig,eval_set_x_orig)
train_set_x_flatten,eval_set_x_flatten = [get_flatten(train_set_x_orig),get_flatten(eval_set_x_orig)]

train_set_x,eval_set_x = [standardize(train_set_x_flatten),standardize(eval_set_x_flatten)]