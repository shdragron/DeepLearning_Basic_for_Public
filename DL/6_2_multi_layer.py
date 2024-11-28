# 6_2_multi_layer.py

import keras
from sklearn import preprocessing, model_selection

mnist = keras.datasets.mnist.load_data()
print(type(mnist),len(mnist))
print(len(mnist[0][0]))
print(len(mnist[1][0]))

(x_train, y_train), (x_test, y_test) = mnist
x_train, x_test = x_train.reshape(-1,784), x_test.reshape(-1,784) # 데이터 개수, feature (이미지의 각점(픽셀) 하나가 feature이다)

x_train, x_test = x_train / 255.0, x_test / 255.0 # 정규화

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(y_train[:10])
print(type(y_train[0]))

model = keras.models.Sequential([

    keras.layers.Input(shape = x_train[0].shape),
    #           데이터 개수    feature      class
    # 1 - layer: (?, 256) = (?,784) @ (784, 256) + 256(bias) 가중치 개수 -> 200,960
    keras.layers.Dense(units=256, activation='relu'),
    #                        feature     class
    # 2 - layer: (?, 64) = (?,256) @ (256, 64) + 64                    -> 16,448
    keras.layers.Dense(units=64, activation='relu'),
    #                        feature    class
    # 2 - layer: (?, 10) = (?,64) @ (64, 10) +10                       ->  650
    keras.layers.Dense(units=10, activation='softmax'),

    # 분류가 중요하다 -> 각각의 레이어의 역할을 주어져야하는데 activation function이 없으면 하나의 방정식으로 진행되기에 분류가 합쳐진다.
])

model.summary() # 모델의 아키텍쳐를 보여준다.

# 모델 컴파일
# SGD: Stochastic gradient descent
# 6만개의 데이터를 다 사용해서 가중칠르 업데이트한다는 것이 매우 비효율적.
# mini batch: 100개
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.01),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
# 모델 훈련
model.fit(x_train, y_train, epochs=10,verbose = 1,
          validation_data=(x_test,y_test),
          batch_size= 100) # 순전파 + 최적화함수 + 역전파


# p = model.predict(x_train)
# print(p)


# 동작을 안한다 입력 데이터가 3차원 우리는 2차원을 넣는다.