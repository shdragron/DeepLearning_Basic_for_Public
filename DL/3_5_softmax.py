# 3_5_softmax.py
import numpy as np


def softmax_1(z):
    s = np.sum(z)
    return z / s


def softmax_2(z):
    z = np.e ** z
    s = np.sum(z)
    return z / s


a = [2.0, -1.0, 0.1]
a = np.array(a)

print(softmax_1(a))
print(softmax_2(a))

# 5가지 중 1가지 선택
# 3번째 표현 -> 0 0 0 1 0
