# 3_2_logistic_regression.py
import keras
import numpy as np

#   시간 출석
x = [[1, 2],
     [2, 1],
     [4, 5],
     [5, 4],
     [8, 9],
     [9, 8]]
y = [[0],           # fail
     [0],
     [1],           # pass
     [1],
     [1],
     [1]]

x = np.array(x)
y = np.array(y)

model = keras.Sequential([
     keras.layers.Input(shape=[2]),          # 데이터 1개의 shape. x[0].shape, x.shape[1:]
     # keras.layers.Dense(1),
     # keras.layers.Activation('sigmoid'),
     keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.binary_crossentropy,
              metrics=['acc'])

model.fit(x, y, epochs=100, verbose=2)

p = model.predict(x, verbose=0)
print(p)

# 퀴즈
# 예측한 결과에 대해 정확도를 구하세요
print(p > 0.5)
print(np.int32(p > 0.5))
print(y == (p > 0.5))
print('acc :', np.mean(y == (p > 0.5)))

# (?, 2) = (?, 4) @ (4, 2)
# (?, 5) = (?, 2) @ (2, 5)
# (?, 3) = (?, 5) @ (5, 3)
