# 3_1_sigmoid.py

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))
# z = (W.T) * X
def show_sigmoid(a):
    a = np.arange(-10,11,1)

    plt.plot(a,sigmoid(a),'b-')
    plt.show()

def log_a():
    return 'a'

def log_b():
    return 'b'

def show_logistic(y):
    print(y * log_a() + (1-y) * log_b())
    # if y == True:
    #     print(log_a())
    # else:
    #     print(log_b())

show_logistic(True)
show_logistic(False)