
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import keras

# 퀴즈: iris.csv 파일에 대해 70%로 학습하고 30%에 대해 정확도를 구하세요.(sparse 버전 사용)
# 의사 코드
# 1. 데이터 읽기
# 2. 모델 구축(레이어 연결)
# 3. 모델 옵션( 손실, 옵티마이저)
# 4. 학습
# 5. 예측
def read_iris():
    df = pd.read_csv('data/iris.csv',delimiter=',',header=0)

    print(df.values)

    x = np.float32(df.values[:,:4])
    y = df.values[:,4]

    print(x.shape)
    print(y.shape)

    x = np.array(x)
    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(y)
    y = np.array(y)

    print(x.shape)
    print(y.shape)
    print(y)

    return x,y

x, y = read_iris()


x = preprocessing.scale(x)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#      (전체)   (데이터 1개) 전달
# 기본: 2차원 -> 1차원 묶음
# RNN: 3차원 -> 2차원 묶음
# CNN: 4차원 -> 3차원 묶음
model = keras.Sequential([
    keras.layers.Input(shape=x[0].shape),
    # keras.layers.Dense(100, activation='softmax'),
    keras.layers.Dense(3, activation='softmax'),
])

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

# 모델 훈련
model.fit(x_train, y_train, epochs = 100,verbose = 1,
          validation_data=(x_test,y_test)) # 순전파 + 최적화함수 + 역전파

print('mae :', model.evaluate(x_test, y_test, verbose=0))