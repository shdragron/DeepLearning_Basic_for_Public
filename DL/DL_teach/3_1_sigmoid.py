# 3_1_sigmoid.py
import matplotlib.pyplot as plt
import numpy as np


# 0 12 34 5.1
# 0 1 0 0 1 1 1
def sigmoid(z):
    return 1 / (1 + np.e ** -z)


def show_sigmoid():
    print('euler :', np.e)
    print(sigmoid(-1))
    print(sigmoid(0))
    print(sigmoid(1))

    # 퀴즈
    # -10에서 10까지의 실수에 대해
    # sigmoid를 통과한 결과를 그래프에 표시하세요
    x = np.linspace(-10, 10)
    for z in x:
        plt.plot(z, sigmoid(z), 'ro')

    plt.plot(x, sigmoid(x))
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

