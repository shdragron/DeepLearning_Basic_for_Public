# 5_1_stock.py
import keras
import numpy as np
from sklearn import preprocessing
import pandas as pd


def make_xy(seq_length):
    stock = np.loadtxt('data/stock_daily.csv', delimiter=',')
    # stock = pd.read_csv('data/stock_daily.csv', skiprows=1)
    # print(stock)

    stock = stock[::-1]     # 날짜 뒤집기

    # stock = preprocessing.scale(stock)
    stock = preprocessing.minmax_scale(stock)

    x = [stock[i:i+seq_length] for i in range(len(stock)-seq_length)]
    y = [stock[i+seq_length, -1:] for i in range(len(stock)-seq_length)]

    return np.array(x), np.array(y)


def rnn_stock_1(seq_length):
    x, y = make_xy(seq_length)
    print(x.shape, y.shape)             # (725, 7, 5) (725, 1)

    model = keras.Sequential([
        keras.layers.Input(shape=x.shape[1:]),
        keras.layers.SimpleRNN(16, return_sequences=False),
        keras.layers.Dense(1)
    ])
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.mean_squared_error,
                  metrics=['mae'])

    model.fit(x, y, epochs=10, verbose=2)


rnn_stock_1(seq_length=7)
