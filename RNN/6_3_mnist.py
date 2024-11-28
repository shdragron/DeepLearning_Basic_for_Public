# 6_3_mnist.py
import keras


# 퀴즈
# mnist 데이터셋에 대해 rnn 모델을 사용해서 결과를 구하세요(lstm 레이어 사용)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# print(x_train.shape)      # (60000, 28, 28)

x_train = x_train / 255
x_test = x_test / 255

model = keras.Sequential([
    keras.layers.Input(shape=x_train.shape[1:]),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
model.summary()

model.compile(optimizer=keras.optimizers.RMSprop(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, verbose=2, batch_size=100,
          validation_data=(x_test, y_test))
