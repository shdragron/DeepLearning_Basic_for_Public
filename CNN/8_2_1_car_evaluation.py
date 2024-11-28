# 8_2_1_car_evaluation.py
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import keras


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
    # exit()

    enc = preprocessing.LabelBinarizer()            # one-hot vector
    y = enc.fit_transform(cars.acceptability)       # (1728, 4)
    y = np.argmax(y, axis=1)                        # (1728,)
    # print(buying)

    x = np.hstack([buying, maint, doors, persons, lug_boot, safety])
    # print(x.shape, y.shape)         # (1728, 21) (1728,)

    return x, y


# x, y = read_car_eval_1()
x, y = read_car_eval_2()

# x = preprocessing.scale(x)
# x = preprocessing.minmax_scale(x)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

model = keras.Sequential()
model.add(keras.layers.Dense(12, activation='relu'))
model.add(keras.layers.Dense(4, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, verbose=2,
          validation_data=(x_test, y_test))
