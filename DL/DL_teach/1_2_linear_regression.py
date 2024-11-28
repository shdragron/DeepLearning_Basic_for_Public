# 1_2_linear_regression.py
import keras
import numpy as np

x = [[1],
     [2],
     [3]]
y = [[1],
     [2],
     [3]]

# x = np.int32(x)
x = np.array(x)
y = np.array(y)
# print(x[0, 0], x[0][0])

# 모델(객체) 생성
model = keras.Sequential()
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
              loss=keras.losses.mean_squared_error,
              metrics=['mae'])

# 학습
model.fit(x, y, epochs=100, verbose=2)   # 0: None, 1: Full, 2: Simple

# 사용
print(model.evaluate(x, y, verbose=0))

p = model.predict(x, verbose=0)
print(p)

# 퀴즈
# 예측한 결과 p를 사용해서 손실을 계산하세요
# c += (hx - y[i]) ** 2
# print(np.mean((p - y) ** 2))

# 퀴즈
# 예측한 결과 p를 사용해서 mae를 계산하세요
print(np.mean(np.abs(p - y)))

