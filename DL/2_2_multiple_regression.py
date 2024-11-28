# 2_2_multiple_regression.py

# 2_1_linear_regression_cars.py
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




y_train =np.array( [[1],
           [2],
           [9],
           [9],
           [17],
           [17]])



# 모델 생성

model = keras.models.Sequential([keras.layers.Dense(units=1)])
# model.add(keras.layers.Dense(units=1))
# 모델 컴파일
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
              loss=keras.losses.MeanSquaredError
              )
# 모델 훈련
model.fit(x_train, y_train, epochs=200,verbose = 1) # 순전파 + 최적화함수 + 역전파


print('this: ',x_train.shape)
sample = np.array([[3, 6],[5, 1]])
p = model.predict(sample) # hx
print(p)
# #
plt.plot(x_train[:,1], y_train, 'yo')
plt.plot(sample[:,1],p,'r-')
plt.plot(x_train[:,0], y_train, 'bo')
plt.plot(sample[:,0],p,'g-')
plt.show()


