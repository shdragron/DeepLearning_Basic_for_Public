# 5_3_softmax.py
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import keras


def read_iris():
    df = pd.read_csv('data/iris_onehot.csv',delimiter=',',header=0)

    print(df.values)

    x = df.values[:,:4]
    y = df.values[:,-3:]

    print(x.shape)
    print(y.shape)

    x = np.array(x)
    y = np.array(y)

    print(x.shape)
    print(y.shape)
    return x,y

x, y = read_iris()

x = preprocessing.scale(x)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#      (전체)   (데이터 1개) 전달
# 기본: 2차원 -> 1차원 묶음
# RNN: 3차원 -> 2차원 묶음
# CNN: 4차원 -> 3차원 묶음
model = keras.Sequential([
    keras.layers.Input(shape=x[0].shape),
    # keras.layers.Dense(100, activation='softmax'),
    keras.layers.Dense(3, activation='softmax'),
])

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.categorical_crossentropy,
              metrics=['acc'])

# 모델 훈련
model.fit(x_train, y_train, epochs = 100,verbose = 1,
          validation_data=(x_test,y_test)) # 순전파 + 최적화함수 + 역전파

print('mae :', model.evaluate(x_test, y_test, verbose=0))
