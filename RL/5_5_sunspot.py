# 5_5_sunspot.py
import numpy as np
import pandas
import pandas as pd
from sklearn import preprocessing, model_selection
import keras
import nltk


# 텐서플로 시험 문제
# 월간 흑점 데이터에 대해 rnn 모델을 만드세요
# 시퀀스 길이: 30개
# 학습: 3000개, 검사: 나머지
# 통과 기준: mae 0.12
df = pd.read_csv('data/monthly-sunspots.csv', index_col=0)
print(df)
print(df.values[:, 0:].shape)               # (2820, 1)

sunspots = preprocessing.minmax_scale(df.values[:, 0:])

grams = nltk.ngrams(sunspots, 31)
grams = list(grams)
print(np.array(grams).shape)                # (2790, 31, 1)

x = np.array([i[:-1] for i in grams])
y = np.array([i[-1] for i in grams])
print(np.array(x).shape)                    # (2790, 30, 1)
print(np.array(y).shape)                    # (2790, 1)

data = model_selection.train_test_split(x, y, train_size=2000)
x_train, x_test, y_train, y_test = data

model = keras.Sequential([
    keras.layers.Input(x.shape[1:]),
    keras.layers.GRU(32, return_sequences=False),
    keras.layers.Dense(1)
])
model.summary()

model.compile(optimizer=keras.optimizers.RMSprop(0.001),
              loss=keras.losses.mean_squared_error,
              metrics=['mae'])

model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_test, y_test))




# "1749-01",58.0
# "1749-02",62.6
# "1749-03",70.0
# "1749-04",55.7
# "1749-05",85.0
# "1749-06",83.5

# ---------------------
# "1749-01",58.0
# "1749-02",62.6
# "1749-03",70.0

# "1749-02",62.6
# "1749-03",70.0
# "1749-04",55.7

# "1749-03",70.0
# "1749-04",55.7
# "1749-05",85.0

# "1749-04",55.7
# "1749-05",85.0
# "1749-06",83.5
