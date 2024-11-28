# 2_3_multiple_regression_trees.py
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt



data = pd.read_csv('data/trees.csv')
# print(data)

x = data.Girth.values.reshape(-1,1)
# print(x)
x_1 = data.Height.values.reshape(-1,1)
# print(x_1)
y = data.Volume.values.reshape(-1,1)

# Girth와 Height 데이터로 Volume을 예측하기

# Girth가 Height가 11, 15 / 80, 90 일때 Volume은?

x = np.hstack([x, x_1])

# x.transpose()

# 모델 생성

model = keras.models.Sequential([keras.layers.Dense(units=1)])
# model.add(keras.layers.Dense(units=1))
# 모델 컴파일
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.5),
              loss=keras.losses.MeanSquaredError,
              metrics = ['RootMeanSquaredError']
              )
# 모델 훈련
model.fit(x, y, epochs = 1000,verbose = 1) # 순전파 + 최적화함수 + 역전파


print('this: ',x.shape)
sample = np.array([[10, 80],[20, 90]])
p = model.predict(sample) # hx
print(p)
# #
plt.plot(x[:,1], y, 'yo')
plt.plot(sample[:,1],p,'r-')
plt.plot(x[:,0], y, 'bo')
plt.plot(sample[:,0],p,'g-')
plt.show()
