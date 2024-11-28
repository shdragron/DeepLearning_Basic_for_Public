# 3_3_logistic_regression_pima.py
import pandas as pd
import numpy as np
import keras
from sklearn import model_selection, preprocessing


# 퀴즈
# 피마 인디언 당뇨병 데이터에 대해
# 70%로 학습하고 30%에 대해 예측하는 딥러닝 모델을 구축하세요
def read_pima():
    pima = pd.read_csv('data/diabetes.csv')
    print(pima)

    x = pima.values[:, :-1]
    y = pima.values[:, -1:]
    print(x.shape, y.shape)

    return x, y


x, y = read_pima()

# x = preprocessing.minmax_scale(x)
x = preprocessing.scale(x)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

# wrong
# x_train = preprocessing.minmax_scale(x_train)
# x_test = preprocessing.minmax_scale(x_test)

model = keras.Sequential([
     keras.layers.Input(shape=x[0].shape),
     keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.binary_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=100, verbose=2)

print('acc :', model.evaluate(x_test, y_test, verbose=0))

p = model.predict(x_test, verbose=0)
print('acc :', np.mean(y_test == (p > 0.5)))

# 퀴즈
# 멀티플 리그레션 trees 모델에 대해 스케일링을 적용해 보세요
