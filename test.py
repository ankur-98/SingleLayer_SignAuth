from std_dataset import *
from pred import *

import cv2
import numpy as np
from tkinter.filedialog import askopenfilename

def getImg(path):
    IMG_SIZE = 100
    img = cv2.imread(path,cv2.IMREAD_ANYCOLOR)
    img = cv2.resize(img, (IMG_SIZE*2, IMG_SIZE), cv2.INTER_LINEAR)
    return img

def test():
    img_path = askopenfilename()
    img = getImg(img_path)
    test_x_org = []
    test_x_org.append(np.array(img))

    test_x_org = np.array(test_x_org)

    test_x_flat = get_flatten(test_x_org)
    test_x = standardize(test_x_flat)

    d = np.load("trained_model5.npy")
    d = d.item()

    w = d["w"]
    b = d["b"]
    train_accuracy = d["train_accuracy"]
    eval_accuracy = d["eval_accuracy"]

    Y = predict(w,b,test_x)

    if Y==0:
        Y = "forged"
    else:
        Y = "genuine"

    print("\n\nThe signature is " + Y + ".")
    print("\n\nThe details of the model used:")
    print("\ntrain accuracy: " + str(train_accuracy))
    print("\neval accuracy: " + str(eval_accuracy))

test()


