# 2_3_functional.py

import keras
import numpy as np

# AND 데이터에 대해 딥러닝 모델을 만드세요.
data = [
    [0, 0, 0,],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
]

# def basic(data):
#     data = np.array(data)
#     print(data.shape)
#     def data_generator(data):
#         x_t, y_t = data[:,:-1], data[:,-1]
#         return x_t, y_t
#
#
#     x_train, y_train = data_generator(data)
#
#     model = keras.models.Sequential([
#
#         keras.layers.Input(shape = x_train[0].shape),
#         keras.layers.Dense(units=1, activation='sigmoid',name='fc_1'),
#
#     ])
#
#     model.summary() # 모델의 아키텍쳐를 보여준다.
#
#
#     model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
#                   loss=keras.losses.BinaryCrossentropy,
#                   metrics=['accuracy'])
#
#     model.fit(x_train, y_train, epochs=20,verbose = 1,
#               batch_size= 100)



def functional(data):
    data = np.array(data)
    print(data.shape)
    def data_generator(data):
        x_t, y_t = data[:,:-1], data[:,-1]
        return x_t, y_t


    x_train, y_train = data_generator(data)

    # 1번

    # input_1 = keras.layers.Input(shape = x_train[0].shape)
    # dense1 = keras.layers.Dense(units=1, activation='relu',name='fc_1')
    # dense2 = keras.layers.Dense(units=1, activation='sigmoid',name='fc_2')
    #
    # output1 = dense1.__call__(input_1)
    # output2 = dense2.__call__(output1) # hx
    #
    #
    # model = keras.models.Model(
    #     inputs=[input_1],outputs=[output2]
    # )

    # 2번 -> 코드 축약 버전

    # input_1 = keras.layers.Input(shape=x_train[0].shape)
    #
    # # output은 한 번씩 중첩되므로 통일
    # output = keras.layers.Dense(units=4, activation='relu').__call__(input_1)
    # output = keras.layers.Dense(units=1, activation='sigmoid').__call__(output)  # hx
    #
    # model = keras.models.Model(
    #     inputs=[input_1], outputs=[output]
    # )

    # 3번 -> 더축약 버전

    input_1 = keras.layers.Input(shape=x_train[0].shape)

    # output은 한 번씩 중첩되므로 통일
    output = keras.layers.Dense(units=4, activation='relu')(input_1)    # __call__ 축약 가능
    output = keras.layers.Dense(units=1, activation='sigmoid')(output)  # __call__ 축약 가능

    model = keras.models.Model(
        inputs=[input_1], outputs=[output]
    )

    model.summary() # 모델의 아키텍쳐를 보여준다.


    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
                  loss=keras.losses.BinaryCrossentropy,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20,verbose = 1,
              batch_size= 100)


def multi_input():
    data = [[0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1]]
    data = np.array(data)

    x1 = data[:, :1]
    x2 = data[:, 1:2]
    y = data[:, 2:]

    input1 = keras.layers.Input(shape=[1])
    output1 = keras.layers.Dense(4, activation='relu')(input1)

    input2 = keras.layers.Input(shape=[1])
    output2 = keras.layers.Dense(4, activation='relu')(input2)

    output = keras.layers.concatenate([output1, output2])
    output = keras.layers.Dense(1, activation='sigmoid')(output)

    model = keras.models.Model([input1, input2], output)

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.binary_crossentropy)

    model.fit([x1, x2], y, epochs=100, verbose=2)
    print(model.predict([x1, x2], verbose=0))


def multi_output():
    data = [[0, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 2, 0]]
    data = np.array(data)

    x = data[:, :2]
    y1 = data[:, 2:3]
    y2 = data[:, 3:]

    input = keras.layers.Input(shape=[2])
    output = keras.layers.Dense(4, activation='relu')(input)

    output1 = keras.layers.Dense(1, activation='sigmoid')(output)
    output2 = keras.layers.Dense(1, activation='sigmoid')(output)

    model = keras.models.Model(input, [output1, output2])

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.binary_crossentropy)

    model.fit(x, [y1, y2], epochs=100, verbose=2)

    p = model.predict(x, verbose=0)
    print(p)

    p_and, p_xor = p
    print(p_and)
    print(p_xor)

def multi_input_and_output():
    data = [[0, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 2, 0]]

    data = np.array(data)

    x1 = data[:, 0]
    x2 = data[:, 1]
    y1 = data[:, 2]
    y2 = data[:, 3]

    input1 = keras.layers.Input(shape=[1])
    output1 = keras.layers.Dense(4, activation='relu')(input1)

    input2 = keras.layers.Input(shape=[1])
    output2 = keras.layers.Dense(4, activation='relu')(input2)

    output = keras.layers.concatenate([output1, output2])
    output1 = keras.layers.Dense(1, activation='sigmoid')(output)
    output2 = keras.layers.Dense(1, activation='sigmoid')(output)

    model = keras.models.Model([input1, input2], outputs=[output1, output2])

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc','acc'])

    model.fit([x1, x2], [y1, y2], epochs=10, verbose=2)

    p = model.predict([x1, x2], verbose=0)
    p_and, p_xor = p
    print(p_and)
    print(p_xor)



multi_input_and_output()