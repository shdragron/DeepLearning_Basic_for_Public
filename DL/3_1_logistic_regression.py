# 3_1_logistic_regression.py


import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# hx = w1 * x1 + w2 * x2 +b
#      1          1       0

#         시간, 출석
x_train = np.array([[1, 2],
           [2, 1],
           [4, 5],
           [5, 4],
           [8, 9],
            [9, 8]])




y_train =np.array( [[0],
                    [0],
                    [1],
                    [1],
                    [1],
                    [1]])



# 모델 생성

model = keras.models.Sequential([
    keras.layers.Input(shape = [2]), # 데이터 한개의 shape -> layer가 아니다.
    # keras.layers.Dense(units=1),
    # keras.layers.Activation('sigmoid'),
    keras.layers.Dense(units=1, activation='sigmoid'),
])

# output    input   active
# (?, 2) = (?, 4) @ (4, 2) layer 1
# (?, 5) = (?, 2) @ (2, 5) layer 2
# (?, 3) = (?, 5) @ (5, 3) layer 3
# ?: 데이터의 개수
# ? 개수를 정하는 것이 불가능

# model.add(keras.layers.Dense(units=1))
# 모델 컴파일
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy']
              )
# 모델 훈련
model.fit(x_train, y_train, epochs=200,verbose = 1) # 순전파 + 최적화함수 + 역전파

p = model.predict(x_train)
print(p)

#   [0.6629096 ]     [0] -> 1애 가깝다: 거짓
#   [0.3110287 ]     [0] -> 0에 가깝다: 참
#   [0.91071165]     [1] -> 1에 가깝다: 참
#   [0.7007252 ]     [1] -> 1에 가깝다: 참
#   [0.98919743]     [1] -> 1에 가깝다: 참
#   [0.9545883 ]     [1] -> 1에 가깝다: 참

# 퀴즈: 예측한 결과에 대한 정확도를 구하세요.

def accuracy(y_true, y_pred):
    t = 0
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    # np.int32(p>0.5)
    for j in range (len(y_pred)):
        if y_pred[j] == y_true[j]:
            t += 1
        else:
            pass
    # np.mean(y == ( p > 0.5 ))
    return t/len(y_pred)

print('acc: ' , accuracy(y_train,p))

# # #
# plt.plot(x_train[:,1], y_train, 'yo')
# plt.plot(sample[:,1],p,'r-')
# plt.plot(x_train[:,0], y_train, 'bo')
# plt.plot(sample[:,0],p,'g-')
# plt.show()
#
#
