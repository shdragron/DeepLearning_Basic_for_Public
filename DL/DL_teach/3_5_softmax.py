# 3_5_softmax.py
import numpy as np


def softmax_1(z):
    return z / np.sum(z)


def softmax_2(z):
    z = np.e ** z
    return z / np.sum(z)


a = [2.0, 1.0, 0.1]
a = np.array(a)

print(softmax_1(a))
print(softmax_2(a))

# 5가지 중에서 1가지 선택
# 3번째 -> 0 0 0 1 0
# 1번째 -> 0 1 0 0 0



