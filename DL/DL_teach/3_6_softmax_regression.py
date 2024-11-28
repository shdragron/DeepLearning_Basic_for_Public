# # 3_6_softmax_regression.py
# import keras
# import numpy as np
#
# #   시간 출석
# x = [[1, 2],
#      [2, 1],
#      [4, 5],
#      [5, 4],
#      [8, 9],
#      [9, 8]]
# y = [[0, 0, 1],           # C
#      [0, 0, 1],
#      [0, 1, 0],           # B
#      [0, 1, 0],
#      [1, 0, 0],           # A
#      [1, 0, 0]]
#
# x = np.array(x)
# y = np.array(y)
#
# model = keras.Sequential([
#      keras.layers.Input(shape=[2]),
#      keras.layers.Dense(3, activation='softmax'),
# ])
#
# model.compile(optimizer=keras.optimizers.SGD(0.1),
#               loss=keras.losses.categorical_crossentropy,
#               metrics=['acc'])
#
# model.fit(x, y, epochs=100, verbose=2)
#
# p = model.predict(x, verbose=0)
# print(p)
#
# # 퀴즈
# # p에 대해 정확도를 구하세요
# y_arg = np.argmax(y, axis=1)
# p_arg = np.argmax(p, axis=1)
# print(y_arg)
# print(p_arg)
#
# print('acc :', np.mean(p_arg == y_arg))
#

print('hello\rworld')

