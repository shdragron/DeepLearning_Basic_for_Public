# 2_4_functional.py

import numpy as np
import keras

# def multi_input_and_output():
#     data = [[0, 0, 0, 0],
#             [0, 1, 0, 1],
#             [1, 0, 0, 1],
#             [1, 1, 2, 0]]
#
#     data = np.array(data)
#
#     x1 = data[:, 0]
#     x2 = data[:, 1]
#     y1 = data[:, 2]
#     y2 = data[:, 3]
#
#     input1 = Input(shape=[1])
#     output1 = Dense(4, activation='relu')(input1)
#
#     input2 = Input(shape=[1])
#     output2 = Dense(4, activation='relu')(input2)
#
#     # output = concatenate([output1, output2])
#     output1 = Dense(1, activation='sigmoid')(output)
#     output2 = Dense(1, activation='sigmoid')(output)
#
#     model = Model([input1, input2], outputs=[output1, output2])
#
#     model.compile(optimizer=keras.optimizers.SGD(0.1),
#                   loss=keras.losses.binary_crossentropy,
#                   metrics=['acc','acc'])
#
#     model.fit([x1, x2], [y1, y2], epochs=10, verbose=2)
#
#     p = model.predict([x1, x2], verbose=0)
#     p_and, p_xor = p
#     print(p_and)
#     print(p_xor)


# 퀴즈: 입력과 출력이 하나지만 중간 레이어들이 다이아몬드 형태인 모델을 만드세요.
def diamond():

    data = [[0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 0]]

    x_test = [[0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0],
            [1, 1]]

    data = np.array(data)
    x_test = np.array(x_test)
    x = data[:, 0:2]
    y = data[:, 2]

    x = x.reshape(-1, 2)
    y = y.reshape(-1, 1)
    x_test = x_test.reshape(-1, 2)

    input1 = keras.layers.Input(shape=[2])
    output = keras.layers.Dense(5, activation='relu', name = '1')(input1)

    output2 = keras.layers.Dense(5, activation='relu', name = '2_1')(output)
    output2 = keras.layers.Dense(16, activation='relu',name = '2_2')(output2)
    output2 = keras.layers.Dense(5, activation='relu',name = '2_3')(output2)

    output4 = keras.layers.Dense(5, activation='relu',name = '3_1')(output)
    output4 = keras.layers.Dense(16, activation='relu',name = '3_2')(output4)
    output4 = keras.layers.Dense(5, activation='relu',name = '3_3')(output4)

    output5 = keras.layers.Dense(5, activation='relu',name = '4_1')(output)
    output5 = keras.layers.Dense(16, activation='relu',name = '4_2')(output5)
    output5 = keras.layers.Dense(5, activation='relu',name = '4_3')(output5)

    output6 = keras.layers.concatenate([output2, output4, output5])
    output6 = keras.layers.Dense(1, activation='sigmoid')(output6)

    model = keras.models.Model(input1, outputs= output6)

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=50, verbose=2)

    model.summary()
    p = model.predict(x_test, verbose=0)
    print(p)


diamond()