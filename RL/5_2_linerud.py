# 5_2_linerud.py
from sklearn import datasets, preprocessing
import keras
import numpy as np


# 퀴즈
# sklearn에 있는 linnerud 데이터셋에 대해 함수형 모델을 구축하세요
# data = datasets.load_linnerud()
# print(data['feature_names'])        # ['Chins', 'Situps', 'Jumps']
# print(data['target_names'])         # ['Weight', 'Waist', 'Pulse']

x, y = datasets.load_linnerud(return_X_y=True)
print(x.shape, y.shape)             # (20, 3) (20, 3)

x = preprocessing.scale(x)

# mean(((?, 3) - (?, 3)) ^ 2)
# mean((hx - y) ^ 2)
# model = keras.Sequential([
#     keras.layers.Dense(16, activation='relu'),
#     keras.layers.Dense(3)
# ])

inputs = keras.layers.Input(shape=x.shape[1:])
output = keras.layers.Dense(16, activation='relu')(inputs)
output = keras.layers.Dense(16, activation='relu')(output)

output1 = keras.layers.Dense(4, activation='relu')(output)
output1 = keras.layers.Dense(1)(output1)

output2 = keras.layers.Dense(4, activation='relu')(output)
output2 = keras.layers.Dense(1)(output2)

output3 = keras.layers.Dense(4, activation='relu')(output)
output3 = keras.layers.Dense(1)(output3)

model = keras.Model(inputs, [output1, output2, output3])

model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.mean_squared_error,
              metrics=['mae', 'mae', 'mae'])

model.fit(x, (y[:, 0:1], y[:, 1:2], y[:, 2:3]), epochs=100, verbose=2, validation_data=(x, y))
