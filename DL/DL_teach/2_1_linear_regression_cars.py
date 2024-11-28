# 2_1_linear_regression_cars.py
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 퀴즈
# cars.csv 파일에 대해 동작하는 딥러닝 모델을 구축하세요
def read_cars_1():
    cars = pd.read_csv('data/cars.csv')
    print(cars)
    print(cars['speed'].values)
    print(cars.speed.values)
    print(cars.speed.values.reshape(-1, 1))

    x = cars.speed.values.reshape(-1, 1)
    y = cars.dist.values.reshape(-1, 1)
    return x, y


def read_cars_2():
    f = open('data/cars.csv', 'r', encoding='utf-8')

    f.readline()

    x, y = [], []
    for line in f:
        # print(line.strip().split(','))
        speed, dist = line.strip().split(',')
        # print(speed, dist)

        # x.append([int(speed)])
        # y.append([int(dist)])
        x.append(int(speed))
        y.append(int(dist))

    print(x)
    f.close()

    # x = np.array(x).reshape(-1, 1)
    return np.reshape(x, [-1, 1]), np.reshape(y, [-1, 1])


x, y = read_cars_1()
# x, y = read_cars_2()
print(x.shape, y.shape)

model = keras.Sequential()
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.SGD(0.001),
              loss=keras.losses.mean_squared_error,
              metrics=['mae'])

model.fit(x, y, epochs=10, verbose=2)

# 퀴즈
# 속도가 30과 50일 때의 제동거리를 구하세요
# p = model.predict(x, verbose=0)
# p = model.predict(np.int32([[4],
#                             [4],
#                             [7],
#                             [7],
#                             [8],
#                             [9]]), verbose=0)

xx = [[0], [30]]
# xx = x
p = model.predict(np.int32(xx), verbose=0)
print(p)

# 퀴즈
# cars.csv 파일을 그래프에 출력하고
# 학습한 결과를 그 위에 표시하세요
plt.plot(x, y, 'ro')
plt.plot(xx, p, 'g')
plt.plot(xx, [0, p[1, 0]], 'k')     # wrong
# plt.plot(xx, p, 'gx')
plt.show()
