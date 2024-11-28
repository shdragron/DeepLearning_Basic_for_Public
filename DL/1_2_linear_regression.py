from tensorflow import keras
import numpy as np
# 데이터
x = np.array([[1],[2],[3]])
y = np.array([[1],[2],[3]])


def mse_loss_function(y_true, y_pred):
    mse = sum((y_true - y_pred) ** 2) / len(y_true)
    return mse

def mae_loss_function(y_true, y_pred):
    mse = sum(abs(y_true - y_pred)) / len(y_true)
    return mse

# 모델 생성

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1))  # Dense 레이어 추가

# 모델 컴파일
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
              loss=keras.losses.MeanSquaredError,
              metrics=['mae'] # mae도 시각화
              )  # optimizers 모듈 사용

# 모델 훈련
model.fit(x, y, epochs=10,verbose = 1) # 순전파 + 최적화함수 + 역전파

# 사용
print()
print(model.evaluate(y, x)) # MSE loss 값

# 테스트

p = model.predict(x) # hx
print(p)

# 퀴즈
# 예측한 결과 p를 사용하여 손실을 계산하세요.

print("MSE: ",mse_loss_function(y, p)) # MSE loss 값

print("MAE: ",mae_loss_function(y, p)) # MAE loss 값











