# 5_1_resnet.py
import keras


# 퀴즈
# resnet34 simple 버전을 만드세요 (함수형 모델 사용)
def resnet_simple_34():
    inputs = keras.layers.Input(shape=[224, 224, 3])
    output = inputs

    # conv 1
    output = keras.layers.Conv2D(64, [7, 7], 2, 'same')(output)
    output = keras.layers.MaxPool2D([2, 2], 2)(output)

    # conv 2
    output = keras.layers.Conv2D(64, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(64, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(64, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(64, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(64, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(64, [3, 3], 1, 'same')(output)

    # conv 3
    output = keras.layers.Conv2D(128, [3, 3], 2, 'same')(output)
    output = keras.layers.Conv2D(128, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(128, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(128, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(128, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(128, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(128, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(128, [3, 3], 1, 'same')(output)

    # conv 4
    output = keras.layers.Conv2D(256, [3, 3], 2, 'same')(output)
    output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)

    # conv 5
    output = keras.layers.Conv2D(512, [3, 3], 2, 'same')(output)
    output = keras.layers.Conv2D(512, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(512, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(512, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(512, [3, 3], 1, 'same')(output)
    output = keras.layers.Conv2D(512, [3, 3], 1, 'same')(output)

    output = keras.layers.GlobalAveragePooling2D()(output)
    output = keras.layers.Dense(1000, activation='softmax')(output)

    model = keras.models.Model(inputs, output)
    model.summary()


# 퀴즈
# 34버전을 반복문으로 변환하세요
def resnet_simple_50():
    inputs = keras.layers.Input(shape=[224, 224, 3])
    output = inputs

    # conv 1
    output = keras.layers.Conv2D(64, [7, 7], 2, 'same')(output)
    output = keras.layers.MaxPool2D([2, 2], 2)(output)

    # conv 2
    for _ in range(3):
        output = keras.layers.Conv2D(64, [1, 1], 1, 'same')(output)
        output = keras.layers.Conv2D(64, [3, 3], 1, 'same')(output)
        output = keras.layers.Conv2D(256, [1, 1], 1, 'same')(output)

    # conv 3
    stride = 2
    for _ in range(4):
        output = keras.layers.Conv2D(128, [1, 1], stride, 'same')(output)
        output = keras.layers.Conv2D(128, [3, 3], 1, 'same')(output)
        output = keras.layers.Conv2D(512, [1, 1], 1, 'same')(output)
        stride = 1

    # conv 4
    stride = 2
    for _ in range(6):
        output = keras.layers.Conv2D(256, [1, 1], stride, 'same')(output)
        output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
        output = keras.layers.Conv2D(1024, [1, 1], 1, 'same')(output)
        stride = 1

    # conv 5
    stride = 2
    for _ in range(3):
        output = keras.layers.Conv2D(512, [1, 1], stride, 'same')(output)
        output = keras.layers.Conv2D(512, [3, 3], 1, 'same')(output)
        output = keras.layers.Conv2D(2048, [1, 1], 1, 'same')(output)
        stride = 1

    # ----------------------
    output = keras.layers.GlobalAveragePooling2D()(output)
    output = keras.layers.Dense(1000, activation='softmax')(output)

    model = keras.models.Model(inputs, output)
    model.summary()


def resnet_50():
    inputs = keras.layers.Input(shape=[224, 224, 3])
    output = inputs

    # conv 1
    output = keras.layers.Conv2D(64, [7, 7], 2, 'same')(output)
    output = keras.layers.MaxPool2D([2, 2], 2)(output)

    # conv 2
    for _ in range(3):
        output = keras.layers.Conv2D(64, [1, 1], 1, 'same')(output)
        output = keras.layers.Conv2D(64, [3, 3], 1, 'same')(output)
        output = keras.layers.Conv2D(256, [1, 1], 1, 'same')(output)

    # conv 3
    # stride = 2
    # for _ in range(4):
    #     output = keras.layers.Conv2D(128, [1, 1], stride, 'same')(output)
    #     output = keras.layers.Conv2D(128, [3, 3], 1, 'same')(output)
    #     output = keras.layers.Conv2D(512, [1, 1], 1, 'same')(output)
    #     stride = 1
    #
    # # conv 4
    # stride = 2
    # for _ in range(6):
    #     output = keras.layers.Conv2D(256, [1, 1], stride, 'same')(output)
    #     output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
    #     output = keras.layers.Conv2D(1024, [1, 1], 1, 'same')(output)
    #     stride = 1
    #
    # # conv 5
    # stride = 2
    # for _ in range(3):
    #     output = keras.layers.Conv2D(512, [1, 1], stride, 'same')(output)
    #     output = keras.layers.Conv2D(512, [3, 3], 1, 'same')(output)
    #     output = keras.layers.Conv2D(2048, [1, 1], 1, 'same')(output)
    #     stride = 1
    #
    # # ----------------------
    # output = keras.layers.GlobalAveragePooling2D()(output)
    # output = keras.layers.Dense(1000, activation='softmax')(output)

    model = keras.models.Model(inputs, output)
    model.summary()


# resnet_simple_34()
resnet_simple_50()
# resnet_50()