# 3_5_ensemble.py
import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection


# 퀴즈
# 레드와인 데이터에 대해
# 70%로 학습하고 30%에 대해 평균오차를 구하세요 (리그레션)
def read_wine(path):
    wine = pd.read_csv(path, delimiter=';')
    # print(wine)

    return wine.values[:, :-1], wine.values[:, -1:]


x, y = read_wine('data/winequality-red.csv')
print(x.shape, y.shape)                 # (1599, 11) (1599, 1)

enc = preprocessing.LabelEncoder()
y = enc.fit_transform(y)

# x = preprocessing.minmax_scale(x)
x = preprocessing.scale(x)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

results = []
for i in range(7):
    model = keras.Sequential()
    # model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(12, activation='relu'))
    model.add(keras.layers.Dense(len(enc.classes_), activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=10, verbose=0,
              validation_data=(x_test, y_test))

    # print(model.evaluate(x_test, y_test, verbose=0))

    p = model.predict(x_test, verbose=0)
    p_arg = np.argmax(p, axis=1)
    print('acc :', np.mean(p_arg == y_test))

    results.append(p)

new_p = np.sum(results, axis=0)
new_p_arg = np.argmax(new_p, axis=1)
print('ensemble :', np.mean(new_p_arg == y_test))

