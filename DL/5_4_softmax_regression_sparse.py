# 5_4_softmax_regression_sparse.py
# onehot의 반대 -> sparse

# 3_6_softmax_regression.py

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#         시간, 출석
x_train = np.array(
          [[1, 2],
           [2, 1],
           [4, 5],
           [5, 4],
           [8, 9],
           [9, 8]])




y_train =np.array( [[0 ,0 ,1],    # c
                    [0 ,0 ,1],
                    [0 ,1 ,0],    # B
                    [0 ,1 ,0],
                    [1 ,0 ,0],    # A
                    [1 ,0 ,0]])

# onehot -> sparse
y_train = np.argmax(y_train,axis=1)





# 모델 생성

model = keras.models.Sequential([
    keras.layers.Input(shape = [2]), # 데이터 한개의 shape -> layer가 아니다.
    # keras.layers.Dense(units=1),
    # keras.layers.Activation('sigmoid'),
    keras.layers.Dense(units=3, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy']
              )
# 모델 훈련
model.fit(x_train, y_train, epochs=200,verbose = 1) # 순전파 + 최적화함수 + 역전파

exit()

p = model.predict(x_train)
print(p)

ac_1 = np.argmax(p,axis=1)
ac_2 = np.argmax(y_train,axis=1)
print(ac_1)
count = 0

for x, y in zip(ac_1, ac_2):
    if x == y:
        count += 1
    else:
        pass
print(count/len(p))




