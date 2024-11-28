# 5_1_boston.py

# 퀴즈: 보스턴 주택가격 데이터에 대해
# 80%로 학습하고 20%에 대해 평균 오차를 알려주세요.

import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
import keras


#         시간, 출석
def read_house():
    House = pd.read_excel('data/BostonHousing.xls')
    print(House)
    x = House.values[:, :-2] # 마지막 1개의 컬럼 제외 시키기
    y = House.values[:, -2] # 마지막에서 2번째 컬럼이 정답,
    y = y.reshape(-1, 1) # y는 2차원이여야한다.

    return x, y

# 정답이 0,1 로지스틱, 여러개 multiple regression

x, y = read_house()
print(x.shape, y.shape)
exit()

x = preprocessing.scale(x)

data = model_selection.train_test_split(x, y, train_size=0.8)
x_train, x_test, y_train, y_test = data


model = keras.models.Sequential([keras.layers.Dense(units=1)])
# model.add(keras.layers.Dense(units=1))
# 모델 컴파일
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
              loss=keras.losses.MeanSquaredError,
              metrics = ['RootMeanSquaredError']
              )
# 모델 훈련
model.fit(x_train, y_train, epochs = 100,verbose = 1) # 순전파 + 최적화함수 + 역전파

print('mae :', model.evaluate(x_test, y_test, verbose=0))




