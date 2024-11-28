# 2_1_linear_regression_cars.py
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 원시적으로 데이터 읽기

def read_file(file_path):
    text = open(file_path, mode='r', encoding='utf-8')

    text.readline() # 위 "speed","dist" 출력해서 뻬기

    x, y = [], []
    for line in text:

        speed, dist = line.strip().split(',') # 공백 삭제 후 ','로 분리 후 리스트에 넣기

        x.append(int(speed))
        y.append(int(dist))

    print(x, y)
    text.close()
    return np.array(x).reshape(-1,1), np.array(y).reshape(-1,1)



dataset = pd.read_csv('data/cars.csv')
# print(dataset.head())
data = pd.DataFrame(dataset)

# print(data.values)



x_train = data.speed.values.reshape(-1,1) # 문제
print(x_train)

y_train = data.dist.values.reshape(-1,1) # 문제 정답
print(y_train)


def mse_loss_function(y_train, y_pred):
    mse = sum((y_train- y_pred) ** 2) / len(y_train)
    return mse

def mae_loss_function(y_train, y_pred):
    mse = sum(abs(y_train- y_pred)) / len(y_train)
    return mse

# 모델 생성

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1))  # Dense 레이어 추가

# 모델 컴파일
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001),
              loss=keras.losses.MeanSquaredError
              )  # optimizers 모듈 사용

# 모델 훈련
model.fit(x_train, y_train, epochs=100,verbose = 1) # 순전파 + 최적화함수 + 역전파

# 사용
print()
print(model.evaluate(y_train, x_train)) # MSE loss 값

# 테스트
# 퀴즈: 속도가 30과 50일 때 제동거리를 구하세요.

p = model.predict(np.array([[0],[30]])) # hx
print(p)

# 퀴즈
# 예측한 결과 p를 사용하여 손실을 계산하세요.

# print("MSE: ",mse_loss_function(y_train, p)) # MSE loss 값
#
# print("MAE: ",mae_loss_function(y_train, p)) # MAE loss 값

# 양질의 데이터가 아니기에 학습하기 어렵다.

plt.plot(x_train, y_train, 'ro')
plt.plot(np.concatenate([[0],[30]]), np.concatenate(p), '-')
plt.show()


p1 = model.predict(np.array([[25],[26]]))

print(p1)
