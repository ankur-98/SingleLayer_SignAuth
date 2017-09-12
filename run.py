from proc import *
from model import *

n = eval(input("\n\nEnter number of iterations: ")) #1000
l = eval(input("Enter learning rate: ")) #0.001
p = bool(eval(input("Enter 1 to display cost else 0: "))) #False

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = n, learning_rate = l, print_cost = p)