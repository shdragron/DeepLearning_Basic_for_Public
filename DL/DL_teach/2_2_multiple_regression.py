# 2_2_multiple_regression.py
import keras
import numpy as np

# hx = w1 * x1 + w2 * x2 + b
#       1         1        0
#   시간 출석
x = [[1, 2],
     [2, 1],
     [4, 5],
     [5, 4],
     [8, 9],
     [9, 8]]
y = [[3],
     [3],
     [9],
     [9],
     [17],
     [17]]

x = np.array(x)
y = np.array(y)

model = keras.Sequential()
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss=keras.losses.mean_squared_error,
              metrics=['mae'])

model.fit(x, y, epochs=10, verbose=2)

# 퀴즈
# 3시간 공부하고 6번 출석한 학생과
# 5시간 공부하고 1번 출석한 학생의 성적을 구하세요
xx = [[3, 6],
      [5, 1]]
p = model.predict(np.int32(xx), verbose=0)
print(p)
