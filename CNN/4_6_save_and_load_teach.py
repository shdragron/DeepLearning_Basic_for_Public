# 4_6_save_load.py
import pandas as pd
import numpy as np
import keras
from sklearn import model_selection, preprocessing


# 퀴즈
# red와 white 와인 데이터에 대해 로지스틱 모델을 구축하세요
# 70%로 학습하고 30%에 대해 정확도를 구하세요
def read_wine(path, label):
    wine = pd.read_csv(path, delimiter=';')
    # print(wine)

    return wine.values, [label] * wine.shape[0]
    # return wine.values, np.zeros(wine.shape[0])


x_red, y_red = read_wine('data/winequality-red.csv', 0)
x_white, y_white = read_wine('data/winequality-white.csv', 1)
print(x_red.shape, len(y_red))

x = np.vstack([x_red, x_white])
y = np.array(y_red + y_white)
y = y[:, np.newaxis]
print(x.shape, y.shape)         # (6497, 12) (6497, 1)

# x = preprocessing.minmax_scale(x)
x = preprocessing.scale(x)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

model = keras.Sequential([
     keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.binary_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, verbose=2,
          validation_data=(x_test, y_test))

print(model.evaluate(x_test, y_test, verbose=0))

model.save('./models/wine.keras')
model.save_weights('./models/wine.weights.h5')

# 가중치 변경
model.fit(x_train, y_train, epochs=10, verbose=0,
          validation_data=(x_test, y_test))

print(model.evaluate(x_test, y_test, verbose=0))

print('-' * 30)
saved_model = keras.models.load_model('./models/wine.keras')
print(saved_model.evaluate(x_test, y_test, verbose=0))

# 처음에 사용하던 모델의 가중치를 저장한 가중치로 변경
model.load_weights('./models/wine.weights.h5')
print(model.evaluate(x_test, y_test, verbose=0))





