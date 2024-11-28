# 3_3_logistic_regression_pima.py

# 퀴즈: 피마 인디언 당뇨병 데이터에 대해 70%는 학습하고 30%에 대해 예측하는 분류를 하세요.

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn import preprocessing


# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

# 데이터 불러오기

data = pd.read_csv('data/diabetes.csv')

db = pd.DataFrame(data)


x_1 = db.Pregnancies.values.reshape(-1,1)
x_2 = db.Glucose.values.reshape(-1,1)
x_3 = db.BloodPressure.values.reshape(-1,1)
x_4 = db.SkinThickness.values.reshape(-1,1)
x_5 = db.Insulin.values.reshape(-1,1)
x_6 = db.BMI.values.reshape(-1,1)
x_7 = db.DiabetesPedigreeFunction.values.reshape(-1,1)
x_8 = db.Age.values.reshape(-1,1)
y = db.Outcome.values.reshape(-1,1)

x = np.hstack((x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8))

# Min_MAx_scaling
# 정규화
# x = preprocessing.minmax_scale(x)

# 표쥰화
x = preprocessing.scale(x)

print(x.shape)

train_num = int(0.7*len(x))

x_train = x[:train_num]
y_train = y[:train_num]
x_test = x[train_num:]
y_test = y[train_num:]

# 아래는 잘못된 것
# x_train = preprocessing.minmax_scale(x_train)
# x_test = preprocessing.minmax_scale(x_test)

# 모델 생성

model = keras.models.Sequential([
    keras.layers.Input(shape = [8]),
    keras.layers.Dense(units=1, activation='sigmoid'),
])

# 모델 컴파일
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy']
              )
# 모델 훈련
model.fit(x_train, y_train, epochs=500,verbose = 1) # 순전파 + 최적화함수 + 역전파

p = model.predict(x_test, verbose = 1)
print(p)

# print(np.int32(p>0.5))
print('acc: ', np.mean(y_test == ( p > 0.5 )))



# 그래프 그리기

# if nrows = 3, ncols = 1, index = 2 -> 312
plt.subplot(10,1,1)
plt.plot(x_1[:train_num],y_train,'ro')
plt.subplot(10,1,2)
plt.plot(x_2[:train_num],y_train,'yo')
plt.subplot(10,1,3)
plt.plot(x_3[:train_num],y_train,'bo')
plt.subplot(10,1,4)
plt.plot(x_4[:train_num],y_train,'go')
plt.subplot(10,1,5)
plt.plot(x_5[:train_num],y_train,'ko')
plt.subplot(10,1,6)
plt.plot(x_6[:train_num],y_train,'o', color='lightcoral')
plt.subplot(10,1,7)
plt.plot(x_7[:train_num],y_train,'o', color='olive')
plt.subplot(10,1,8)
plt.plot(x_8[:train_num],y_train,'o', color='midnightblue')
plt.subplot(10,1,9)
plt.plot(x_test,p,'g-')
plt.subplot(10,1,10)
plt.plot(x_1,x_2,'bo')

plt.show()



# 결론: 숫자가 작은 데이터의 변화는 결과에 영향을 많이끼친다.

# 이 방법을 해결하기 위해 scaling을 진행

# scaling: 표준화, 정규화

# MIN - MAX Sacling

# 20 ~ 50 = 범위 30
# 30 - 20 = 10 -> 10/30 -> 33.3333.. %

# 표준화

# (x - 평균) / 표준편차
















