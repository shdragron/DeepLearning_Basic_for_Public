# 9_1_OneHotEncoder.py
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import keras
import tensorflow as tf


# 퀴즈
# 자동차 데이터에 대해 70%로 학습하고 30%에 대해 정확도를 구하세요
def read_car_eval_1():
    names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']
    cars = pd.read_csv('data/car.data', names=names)
    # print(cars)

    enc = preprocessing.LabelEncoder()
    buying = enc.fit_transform(cars.buying)
    maint = enc.fit_transform(cars.maint)
    doors = enc.fit_transform(cars.doors)
    persons = enc.fit_transform(cars.persons)
    lug_boot = enc.fit_transform(cars.lug_boot)
    safety = enc.fit_transform(cars.safety)
    y = enc.fit_transform(cars.acceptability)
    # print(buying)

    x = [buying, maint, doors, persons, lug_boot, safety]
    x = np.transpose(x)
    # print(x)
    # print(x.shape, y.shape)         # (1728, 6) (1728,)

    return x, y


def read_car_eval_2():
    names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']
    cars = pd.read_csv('data/car.data', names=names)

    enc = preprocessing.LabelBinarizer()            # one-hot vector
    buying = enc.fit_transform(cars.buying)
    maint = enc.fit_transform(cars.maint)
    doors = enc.fit_transform(cars.doors)
    persons = enc.fit_transform(cars.persons)
    lug_boot = enc.fit_transform(cars.lug_boot)
    safety = enc.fit_transform(cars.safety)

    # enc = preprocessing.OneHotEncoder()
    # x = enc.fit_transform(cars.values[:, :-1])
    # print(x[0])
    # print(x.shape)
    # print(type(x))

    enc = preprocessing.LabelBinarizer()            # one-hot vector
    y = enc.fit_transform(cars.acceptability)       # (1728, 4)
    y = np.argmax(y, axis=1)                        # (1728,)
    # print(buying)

    x = np.hstack([buying, maint, doors, persons, lug_boot, safety])
    # print(x.shape, y.shape)         # (1728, 21) (1728,)

    return x, y


# get_dummies
def read_car_eval_3():
    names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']
    cars = pd.read_csv('data/car.data', names=names)

    dummies = pd.get_dummies(cars)
    print(dummies.shape)                # (1728, 25)
    print(dummies.columns)
    # ['buying_high', 'buying_low', 'buying_med', 'buying_vhigh', 'maint_high',
    # 'maint_low', 'maint_med', 'maint_vhigh', 'doors_2', 'doors_3', 'doors_4',
    # 'doors_5more', 'persons_2', 'persons_4', 'persons_more', 'lug_boot_big',
    # 'lug_boot_med', 'lug_boot_small', 'safety_high', 'safety_low', 'safety_med',
    # 'acceptability_acc', 'acceptability_good', 'acceptability_unacc', 'acceptability_vgood']

    # 퀴즈
    # get_dummies 함수로 만든 데이터에 대해 동작하도록 x, y를 만드세요
    print(dummies.values)
    print(dummies.values.dtype)
    print(np.int32(dummies.values))

    src = np.int32(dummies.values)
    x = src[:, :-4]
    y = src[:, -4:]
    y = np.argmax(y, axis=1)

    return x, y


# get_dummies
def read_car_eval_4():
    names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']
    cars = pd.read_csv('data/car.data', names=names)

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(cars['acceptability'])
    cars = cars.drop(['acceptability'], axis=1)

    dummies = pd.get_dummies(cars)
    print(dummies.shape)                # (1728, 21)
    print(dummies.columns)

    print(dummies.values)
    print(dummies.values.shape)
    print(dummies.values.dtype)
    print(np.int32(dummies.values))

    x = np.int32(dummies.values)

    # dummies = pd.get_dummies(cars, sparse=True)
    # print(dummies.shape)                # (1728, 25)
    # print(dummies.columns)
    # print(dummies)
    # print(dummies.values)

    return x, y


# OneHotEncoder
def read_car_eval_5():
    names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']
    cars = pd.read_csv('data/car.data', names=names)

    enc = preprocessing.OneHotEncoder(sparse_output=True)
    x = enc.fit_transform(cars.values[:, :-1])
    print(x[0])                             # (0, 3)	1.0
    print(x.shape)                          # (1728, 21)
    print(type(x))                          # <class 'scipy.sparse._csr.csr_matrix'>

    enc = preprocessing.OneHotEncoder(sparse_output=False)
    x = enc.fit_transform(cars.values[:, :-1])
    print(x[0])                             # [0. 0. 0. 1. 0. ...]
    print(x.shape)                          # (1728, 21)
    print(type(x))                          # <class 'numpy.ndarray'>

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(cars.acceptability)

    return x, y


# x, y = read_car_eval_1()
# x, y = read_car_eval_2()
# x, y = read_car_eval_3()
# x, y = read_car_eval_4()
x, y = read_car_eval_5()

# OneHotEncoder, LabelBinarizer 결과 비교
# x1, _ = read_car_eval_2()
# x2, _ = read_car_eval_5()
# print(x1 == x2)
# print(np.all(x1 == x2))
#
# for i, j in zip(x1.flatten(), x2.flatten()):
#     assert i == j

# x = preprocessing.scale(x)
# x = preprocessing.minmax_scale(x)

# data = model_selection.train_test_split(x, y, train_size=0.7)
data = model_selection.train_test_split(x, y, train_size=0.7, shuffle=False)
x_train, x_test, y_train, y_test = data

np.random.seed(21)

model = keras.Sequential()
# model.add(keras.layers.Dense(12, activation='relu', kernel_initializer='ones'))
model.add(keras.layers.Dense(12, activation='relu'))
model.add(keras.layers.Dense(4, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, verbose=2,
          validation_data=(x_test, y_test), shuffle=False)
