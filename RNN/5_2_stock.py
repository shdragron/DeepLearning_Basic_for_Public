# 5_2_stock.py
import keras
import numpy as np
from sklearn import preprocessing
import csv
import pandas as pd
import matplotlib.pyplot as plt


# 퀴즈
# GOOG.csv 파일에 대해 동작하는 RNN 모델을 만드세요
def make_xy(seq_length):
    # 1번
    # f = open('data/GOOG.csv', 'r', encoding='utf-8')
    # f.readline()
    #
    # stock = []
    # for row in csv.reader(f):
    #     # print(row[1], row[2], row[3], row[6], row[4])
    #     stock.append((float(row[1]), float(row[2]), float(row[3]), float(row[6]), float(row[4])))
    #
    # f.close()

    # 2번
    df = pd.read_csv('data/GOOG.csv', index_col=0)
    # print(df)

    stock = [df.Open, df.High, df.Low, df.Volume, df.Close]
    stock = np.array(stock)
    print(stock)
    print(stock.shape)              # (5, 252)

    stock = stock.transpose()
    print(stock.shape)              # (252, 5)

    scaler = preprocessing.MinMaxScaler()
    stock = scaler.fit_transform(stock)

    stock = scaler.inverse_transform(stock)

    # stock = preprocessing.scale(stock)
    # stock = preprocessing.minmax_scale(stock)

    x = [stock[i:i + seq_length] for i in range(len(stock) - seq_length)]
    y = [stock[i + seq_length, -1:] for i in range(len(stock) - seq_length)]

    return np.array(x), np.array(y)


def rnn_stock_2(seq_length):
    x, y = make_xy(seq_length)
    print(x.shape, y.shape)     # (245, 7, 5) (245, 1)

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

    # 퀴즈
    # x에 대해 예측한 결과와 정답을 그래프로 그려주세요
    p = model.predict(x, verbose=0)
    print(p.shape, y.shape)         # (245, 1) (245, 1)

    plt.plot(range(len(p)), p, 'r', label='pred')
    plt.plot(range(len(p)), y, 'g', label='target')
    plt.legend()
    plt.show()


rnn_stock_2(seq_length=7)
