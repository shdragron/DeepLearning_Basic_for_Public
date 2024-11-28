# 4_5_bank.py
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
import keras


# 퀴즈
# 신규고객 유치 영업 활동 데이터에 대해서
# 80%로 학습하고 20%에 대해 정확도를 계산하는 딥러닝 모델을 구축하세요
def read_bank():
    bank = pd.read_csv('data/bank.csv', delimiter=';')
    print(bank)

    enc = preprocessing.LabelEncoder()

    # 1번
    # job = enc.fit_transform(bank.job)
    # marital = enc.fit_transform(bank.marital)
    # print(job)
    # print(marital)
    #
    # binds = [job, marital, bank.age]
    # binds = np.array(binds).T
    # print(binds)

    binds = []
    for name in ['job', 'marital', 'education', 'default']:
        binds.append(enc.fit_transform(bank[name]))

    binds.append(bank.age)
    binds = np.array(binds).T

    y = enc.fit_transform(bank.y)
    return binds, y.reshape(-1, 1)


x, y = read_bank()
print(x.shape, y.shape)         # (4521, 3) (4521, 1)

x = preprocessing.scale(x)
# x = preprocessing.minmax_scale(x)

data = model_selection.train_test_split(x, y, train_size=0.8)
x_train, x_test, y_train, y_test = data

model = keras.Sequential([
    keras.layers.Input(shape=x[0].shape),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.binary_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=100, verbose=2)
print('acc :', model.evaluate(x_test, y_test, verbose=0))
