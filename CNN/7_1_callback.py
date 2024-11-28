# 7_1_callback.py
import keras
import pandas as pd
from sklearn import preprocessing, model_selection


def read_wine(path):
    wine = pd.read_csv(path, delimiter=';')
    return wine.values[:, :-1], wine.values[:, -1]


x, y = read_wine('data/winequality-red.csv')

enc = preprocessing.LabelEncoder()
y = enc.fit_transform(y)

x = preprocessing.scale(x)

data = model_selection.train_test_split(x, y, train_size=0.7, shuffle=True)
x_train, x_test, y_train, y_test = data


# class MyModel(keras.Sequential):
#     def train_step(self, data):
#         print('train_step')


model = keras.Sequential()
model.add(keras.layers.Dense(12, activation='relu'))
model.add(keras.layers.Dense(len(enc.classes_), activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])


class Custom(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.count = 0
    
    # 테스트 에포크가 끝났을 때 알려주는 코드를 넣어보세요
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2:
            print('epoch begin', epoch, logs)

    def on_test_batch_begin(self, batch, logs=None):
        print('테스트 배치 시작', batch)
        self.count += 1

    def on_train_end(self, logs=None):
        print('테스트 종료', logs)


custom = Custom()

model.fit(x_train, y_train, epochs=10, verbose=0, callbacks=[custom],
          validation_data=(x_test, y_test),batch_size=10, shuffle=True)

print('count :', custom.count)
