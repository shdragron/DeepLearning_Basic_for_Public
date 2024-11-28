# 4_3_raisin.py
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
import keras


# 퀴즈
# 건포도 데이터셋에 대해
# 60% 데이터로 학습하고 20% 데이터에 대해 검증하고
# 최종적으로 마지막 20%로 검사하는 딥러닝 모델을 구축하세요
def read_raisin():
    raisin = pd.read_excel('data/Raisin_Dataset.xlsx')
    # print(raisin)

    x = np.float32(raisin.values[:, :-1])
    y = raisin.values[:, -1:]

    enc = preprocessing.LabelEncoder().fit(y)
    y = enc.transform(y)

    return x, y.reshape(-1, 1)


x, y = read_raisin()
print(x.shape, y.shape)         # (900, 7) (900, 1)

x = preprocessing.scale(x)
# x = preprocessing.minmax_scale(x)

data = model_selection.train_test_split(x, y, train_size=0.6)
x_train, x_other, y_train, y_other = data

data = model_selection.train_test_split(x_other, y_other, train_size=0.5)
x_valid, x_test, y_valid, y_test = data

model = keras.Sequential([
    keras.layers.Input(shape=x[0].shape),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.binary_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=100, verbose=2)

p = model.predict(x_valid, verbose=0)
print('acc :', np.mean(x_valid == (p > 0.5)))

print('acc :', model.evaluate(x_test, y_test, verbose=0))
