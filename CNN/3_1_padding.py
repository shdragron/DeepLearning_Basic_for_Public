# 3_1_padding.py
import  tensorflow as tf
import numpy as np


a1 = np.arange(0, 4, 1) + 1
a2 = np.arange(0, 12, 1) .reshape(3, 4) + 1
a3 = np.arange(0, 24, 1) .reshape(2, 3, 4) + 1

print(a1)
print(np.pad(a1,1))
print('-'*30)
print(a2)
print(np.pad(a2,1))
print(np.pad(a2,([1,2],[1,2])))

print('-'*30)
print(a3)
print(a3.shape)
print(np.pad(a3,1,'constant',constant_values=0))
print(np.pad(a3,1,'constant',constant_values=0).shape)
print('-'*30)
print(np.pad(a3,([0,0],[1,1],[1,1])))



# tf.pad(t, paddings, "CONSTANT") # zero padding