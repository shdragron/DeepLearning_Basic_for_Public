# 6_2_upsampling.py

import keras
import numpy as np


input_shape=( 2, 2, 1, 3)
x = np.arange(np.prod(input_shape)).reshape(input_shape)
print(x.shape)

y = keras.layers.UpSampling2D(size = [1, 2])(x)
print(y.shape)

y = keras.layers.UpSampling2D(size = [2, 1])(x)
print(y.shape)













