# 4_6_saveandload.py

# 퀴즈: red, wine 데이터에 대해 로지스틱 모델을 구축하세요.
# 70% 학습 30% 정확도
import numpy as np
import pandas as pd
import keras
from sklearn import model_selection, preprocessing

data_1 = pd.read_csv('data/winequality-red.csv',sep=';')
data_2 = pd.read_csv('data/winequality-white.csv',sep=';')

data = np.concatenate([data_1.values,data_2.values],axis=0)

print(data_1.values.shape)
print(data_2.values.shape)
print(data.shape)

data = preprocessing.scale(data)

y_train = []
y_test = []

for i in range(1599):
    y_train.append(0)
for i in range(4898):
    y_test.append(1)
y_train = np.array(y_train)
y_test = np.array(y_test)
y = np.concatenate([y_train,y_test])


data_3 = model_selection.train_test_split(data, y, train_size=0.7,shuffle=True)

x_train, x_test, y_train, y_test = data_3

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(x_test)

model = keras.Sequential([
    keras.layers.Dense(units=1, activation='sigmoid', name='fc_1'),

])

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.binary_crossentropy,
              metrics=['acc'])

# model.fit(flow_train, epochs=10, verbose=1)

model.fit(x = x_train,y = y_train, epochs=10, verbose=1,validation_data=(x_test,y_test))

# print(model.evaluate(x_test,y_test))

model.save('./models/wine.keras')
model.save_weights('./models/wine.weights.h5')

saved_model = keras.models.load_model('./models/wine.keras')
print(saved_model.evaluate(x_test,y_test))

model.load_weights('./models/wine.weights.h5')

print(model.evaluate(x_test,y_test))










































