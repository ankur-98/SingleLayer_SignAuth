from proc import *
from model import *
# from test import *

import numpy as np
import matplotlib.pyplot as plt

#n = eval(input("\n\nEnter number of iterations: ")) #1000
#l = eval(input("Enter learning rate: ")) #0.001
#p = bool(eval(input("Enter 1 to display cost else 0: "))) #False

n=6000
l=0.0001971
p=0

d = model(train_set_x, train_set_y, eval_set_x, eval_set_y, num_iterations = n, learning_rate = l, print_cost = p)

#retrieving essential data about model from dictionary 'd'
costs = d["costs"]
learning_rate = d["learning_rate"]

#saving the trained model for future use for evaling
np.save('trained_model12.npy',d)

#ploting cost vs iteration graph
plt.plot((costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens) ')
plt.title("Learning rate = " + str(learning_rate))
plt.show()

# test()