# 4_2_wdbc.py
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
import keras


# 유방암 데이터에 대해
# 70%로 학습하고 30%에 대해 예측하는 딥러닝 모델을 구축하세요
def read_wdbc():
    wdbc = pd.read_csv('data/wdbc.data',
                       header=None, index_col=0)
    print(wdbc)

    x = wdbc.values[:, 1:]
    x = np.float32(x)
    y = wdbc.values[:, 0]
    print(x.shape, x.dtype, y.shape)
    print(y[:5])

    enc = preprocessing.LabelEncoder()
    enc.fit(y)
    y = enc.transform(y)
    print(y[:5])
    print(enc.classes_)

    return x, y.reshape(-1, 1)


x, y = read_wdbc()

x = preprocessing.scale(x)
# x = preprocessing.minmax_scale(x)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

model = keras.Sequential([
    keras.layers.Input(shape=x[0].shape),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.binary_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=100, verbose=2)

p = model.predict(x_test, verbose=0)
print('acc :', np.mean(y_test == (p > 0.5)))
