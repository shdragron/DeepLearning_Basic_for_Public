# 8_2_car_evaluation.py

# 퀴즈: 자동차 데이터에 대해 70%로 학습하고 30%에대해 정확도를 구하세요.

import pandas as pd
import keras
from sklearn import preprocessing, model_selection

data = pd.read_csv('./data/car.data',header=None)
# print(data.values)

num = round((len(data.values)/100) * 70)

enc = preprocessing.LabelEncoder()
enc2 = preprocessing.OneHotEncoder()
enc3 = preprocessing.LabelBinarizer()
preprocessing.MultiLabelBinarizer

x = data.values[:,:-1]
x = enc2.fit_transform((x))
# x = enc3.fit_transform(x)
# x = preprocessing.scale(x)

print(x[0])

print(x.shape)


y = enc.fit_transform(data.values[:,-1])
y = enc.fit_transform(y)


data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data


print(x_train[0].shape, y_train.shape)
print(x_test[0].shape, y_test.shape)


model = keras.models.Sequential([
    keras.layers.Input(shape=([21])),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='softmax'),
])
model.summary()
model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

ans = model.fit(x_train,y_train,epochs=50, validation_data=(x_test,y_test))

p = model.predict(x_test)
print(p,y_test)


print(ans)



