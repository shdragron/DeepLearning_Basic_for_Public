# 6_4_callback.py
import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt


def read_wine(path):
    wine = pd.read_csv(path, delimiter=';')
    return wine.values[:, :-1], wine.values[:, -1]


x, y = read_wine('data/winequality-red.csv')

enc = preprocessing.LabelEncoder()
y = enc.fit_transform(y)

x = preprocessing.scale(x)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data


model = keras.Sequential()
model.add(keras.layers.Dense(12, activation='relu'))
model.add(keras.layers.Dense(len(enc.classes_), activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1)
# checkpoints = keras.callbacks.ModelCheckpoint('./models/{epoch:02d}-{val_acc:.2f}.keras', monitor='val_acc', verbose=1,
#                                               save_best_only=True)


plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', # 수렴한거 아니야? LR: learning rate * factor(줄이는 스케일)
                                  factor=0.1,
                                  patience=5,
                                  verbose=1)


history = model.fit(x_train, y_train, epochs=300, verbose=1, validation_data=(x_test, y_test),
                    callbacks=[early_stopping, plateau])


def plot_history_1(history):

    print(type(history))
    print(history.history.keys())
    print(history.history['acc'])
    # dict_keys(['acc', 'loss', 'val_acc', 'val_loss'])

    # 퀴즈: 정확도와 손실을 그래프에 그려주세요.

    size = len(history.history['acc'])
    plt.subplot(1, 2, 1)
    plt.plot(range(size),history.history['val_acc'], color='blue', label='test')
    plt.plot(range(size),history.history['acc'], color='red', label='train')
    plt.title('accuracy')
    plt.legend(['test_acc', 'train_acc'])
    plt.subplot(1, 2, 2)
    plt.plot(range(size),history.history['val_loss'], color='blue', label='test')
    plt.plot(range(size),history.history['loss'], color='red', label='train')
    plt.title('loss')
    plt.legend(['test_loss', 'train_loss'])
    plt.tight_layout()
    plt.show()


plot_history_1(history)


