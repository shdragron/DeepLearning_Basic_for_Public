# 2_3_multiple_regression_trees.py
import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing


# 퀴즈
# trees.csv 파일에 대해서
# Girth, Height가 x, Volume을 y로 학습하세요
# 그리고
# Girth가 11, Height가 80일 때와
# Girth가 15, Height가 90일 때의 Volume을 구하세요
def read_trees_1():
    trees = pd.read_csv('data/trees.csv', index_col=0)
    print(trees)
    print(trees.values)

    return trees.values[:, :-1], trees.values[:, -1:]


def read_trees_2():
    trees = pd.read_csv('data/trees.csv', index_col=0)

    girth = trees.Girth.values          # (31,)
    height = trees.Height.values
    volume = trees.Volume.values.reshape(-1, 1)
    print(girth)

    # x = np.array([girth, height])
    # x = x.transpose()

    # hstack([(31, 1), (31, 1)])
    x = np.hstack([girth.reshape(-1, 1), height.reshape(-1, 1)])
    print(x.shape)

    return x, volume


x, y = read_trees_1()
# x, y = read_trees_2()
print(x.shape, y.shape)

# x = preprocessing.minmax_scale(x)
x = preprocessing.scale(x)

model = keras.Sequential()
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss=keras.losses.mean_squared_error,
              metrics=['mae'])

model.fit(x, y, epochs=100, verbose=2)

xx = [[11, 80],
      [15, 90]]
p = model.predict(np.int32(xx), verbose=0)
print(p)

# (2, 9) = (2, 3) @ (3, 9)

















