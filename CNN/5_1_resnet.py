# 5_1_resnet.py

import keras

# resnet34 함수형으로 만드세요.
def resnet_simple_34():

    input = keras.layers.Input(shape=(224, 224, 3))

    # conv 1
    output = keras.layers.Conv2D(64, (7, 7),strides= 2, padding= 'same', name = '1')(input)
    pool = keras.layers.MaxPooling2D(pool_size= 2, strides= 2, padding= 'same',name = 'pool_1')(output)

    # conv 2
    output1 = keras.layers.Conv2D(64, (3, 3), strides= 1, padding= 'same',name = '2')(pool)
    output1 = keras.layers.Conv2D(64, (3, 3), strides= 1, padding= 'same',name = '3')(output1)
    output1 = keras.layers.Conv2D(64, (3, 3), strides= 1, padding= 'same',name = '4')(output1)
    output1 = keras.layers.Conv2D(64, (3, 3), strides= 1, padding= 'same',name = '5')(output1)
    output1 = keras.layers.Conv2D(64, (3, 3), strides= 1, padding= 'same',name = '6')(output1)
    output1 = keras.layers.Conv2D(64, (3, 3), strides= 1, padding= 'same',name = '7')(output1)

    output2 = keras.layers.Conv2D(128, (3, 3), strides= 2, padding= 'same',name = '8')(output1)

    output2 = keras.layers.Conv2D(128, (3, 3), strides= 1, padding= 'same',name = '9')(output2)
    output2 = keras.layers.Conv2D(128, (3, 3), strides= 1, padding= 'same',name = '10')(output2)
    output2 = keras.layers.Conv2D(128, (3, 3), strides= 1, padding= 'same',name = '11')(output2)
    output2 = keras.layers.Conv2D(128, (3, 3), strides= 1, padding= 'same',name = '12')(output2)
    output2 = keras.layers.Conv2D(128, (3, 3), strides= 1, padding= 'same',name = '13')(output2)
    output2 = keras.layers.Conv2D(128, (3, 3), strides= 1, padding= 'same',name = '14')(output2)
    output2 = keras.layers.Conv2D(128, (3, 3), strides= 1, padding= 'same',name = '15')(output2)

    output3 = keras.layers.Conv2D(256, (3, 3), strides=2, padding='same', name='16')(output2)

    output3 = keras.layers.Conv2D(256, (3, 3), strides= 1, padding= 'same',name = '17')(output3)
    output3 = keras.layers.Conv2D(256, (3, 3), strides= 1, padding= 'same',name = '18')(output3)
    output3 = keras.layers.Conv2D(256, (3, 3), strides= 1, padding= 'same',name = '19')(output3)
    output3 = keras.layers.Conv2D(256, (3, 3), strides= 1, padding= 'same',name = '20')(output3)
    output3 = keras.layers.Conv2D(256, (3, 3), strides= 1, padding= 'same',name = '21')(output3)
    output3 = keras.layers.Conv2D(256, (3, 3), strides= 1, padding= 'same',name = '22')(output3)
    output3 = keras.layers.Conv2D(256, (3, 3), strides= 1, padding= 'same',name = '23')(output3)
    output3 = keras.layers.Conv2D(256, (3, 3), strides= 1, padding= 'same',name = '24')(output3)
    output3 = keras.layers.Conv2D(256, (3, 3), strides= 1, padding= 'same',name = '25')(output3)
    output3 = keras.layers.Conv2D(256, (3, 3), strides= 1, padding= 'same',name = '26')(output3)
    output3 = keras.layers.Conv2D(256, (3, 3), strides= 1, padding= 'same',name = '27')(output3)


    output4 = keras.layers.Conv2D(512, (3, 3), strides=2, padding='same', name='28')(output3)

    output4 = keras.layers.Conv2D(512, (3, 3), strides= 1, padding= 'same',name = '31')(output4)
    output4 = keras.layers.Conv2D(512, (3, 3), strides= 1, padding= 'same',name = '32')(output4)
    output4 = keras.layers.Conv2D(512, (3, 3), strides= 1, padding= 'same',name = '33')(output4)
    output4 = keras.layers.Conv2D(512, (3, 3), strides= 1, padding= 'same',name = '34')(output4)
    output4 = keras.layers.Conv2D(512, (3, 3), strides= 1, padding= 'same',name = '35')(output4)

    output4 = keras.layers.GlobalAvgPool2D(name = '36')(output4)
    # output4 = keras.layers.Flatten()(output4)
    output5 = keras.layers.Dense(1000, activation='softmax')(output4)

    model = keras.models.Model(inputs=input, outputs=output5)

    model.summary()

# 반복문으로 전환
def resnet_simple_50():

    input = keras.layers.Input(shape=(224, 224, 3))

    # conv 1
    output = keras.layers.Conv2D(64, (7, 7),strides= 2, padding= 'same',)(input)
    output = keras.layers.MaxPooling2D(pool_size= 2, strides= 2, padding= 'same',)(output)

    # conv 2
    for _ in range(3):
        output = keras.layers.Conv2D(64, (1, 1), strides= 1, padding= 'same',)(output)
        output = keras.layers.Conv2D(64, (3, 3), strides= 1, padding= 'same',)(output)
        output = keras.layers.Conv2D(256, (1, 1), strides= 1, padding= 'same',)(output)

    # conv 3
    stride = 2
    for _ in range(4):
        output = keras.layers.Conv2D(128, (1, 1), strides= stride, padding= 'same')(output)
        output = keras.layers.Conv2D(128, (3, 3), strides= 1, padding= 'same')(output)
        output = keras.layers.Conv2D(512, (1, 1), strides= 1, padding= 'same')(output)
        stride = 1
    # conv 4
    stride = 2
    for _ in range(6):
        output = keras.layers.Conv2D(256, (1, 1), strides=stride, padding='same')(output)
        output = keras.layers.Conv2D(256, (3, 3), strides=1, padding='same')(output)
        output = keras.layers.Conv2D(1024, (1, 1), strides=1, padding='same')(output)
        stride = 1

    # conv 5
    stride = 2
    for i in range(3):
        output = keras.layers.Conv2D(512, (1, 1), strides=stride, padding='same')(output)
        output = keras.layers.Conv2D(512, (3, 3), strides=1, padding='same')(output)
        output = keras.layers.Conv2D(2048, (1, 1), strides=1, padding='same')(output)
        stride = 1
    # ------------------------------------------------------------------------------------------------
    output = keras.layers.GlobalAvgPool2D(name = '36')(output)
    # output4 = keras.layers.Flatten(name = 'Flatten')(output)
    output5 = keras.layers.Dense(1000, activation='softmax')(output)
    model = keras.models.Model(inputs=input, outputs=output5)

    model.summary()



def resnet_50():
    inputs = keras.layers.Input(shape=[224, 224, 3])
    output = inputs

    # conv 1
    output = keras.layers.Conv2D(64, [7, 7], 2, 'same')(output)
    output = keras.layers.MaxPool2D([2, 2], 2)(output)

    # conv 2
    # 퀴즈
    # shortcut connection을 추가하세요
    shortcut = keras.layers.Conv2D(256, [1, 1], 1, 'same')(output)

    for _ in range(3):
        output = keras.layers.Conv2D(64, [1, 1], 1, 'same')(output)
        output = keras.layers.Conv2D(64, [3, 3], 1, 'same')(output)
        output = keras.layers.Conv2D(256, [1, 1], 1, 'same')(output)

        output = keras.layers.add([output, shortcut])
        shortcut = output

    # conv 3
    shortcut = keras.layers.Conv2D(512, [1, 1], 2, 'same')(output)

    stride = 2
    for _ in range(4):
        output = keras.layers.Conv2D(128, [1, 1], stride, 'same')(output)
        output = keras.layers.Conv2D(128, [3, 3], 1, 'same')(output)
        output = keras.layers.Conv2D(512, [1, 1], 1, 'same')(output)

        output = keras.layers.add([output, shortcut])
        shortcut = output
        stride = 1

    # conv 4
    shortcut = keras.layers.Conv2D(1024, [1, 1], 2, 'same')(output)

    stride = 2
    for _ in range(6):
        output = keras.layers.Conv2D(256, [1, 1], stride, 'same')(output)
        output = keras.layers.Conv2D(256, [3, 3], 1, 'same')(output)
        output = keras.layers.Conv2D(1024, [1, 1], 1, 'same')(output)

        output = keras.layers.add([output, shortcut])
        shortcut = output
        stride = 1

    # conv 5
    shortcut = keras.layers.Conv2D(2048, [1, 1], 2, 'same')(output)

    stride = 2
    for _ in range(3):
        output = keras.layers.Conv2D(512, [1, 1], stride, 'same')(output)
        output = keras.layers.Conv2D(512, [3, 3], 1, 'same')(output)
        output = keras.layers.Conv2D(2048, [1, 1], 1, 'same')(output)

        output = keras.layers.add([output, shortcut])
        shortcut = output
        stride = 1

    # ----------------------
    output = keras.layers.GlobalAveragePooling2D()(output)
    output = keras.layers.Dense(1000, activation='softmax')(output)

    model = keras.models.Model(inputs, output)
    model.summary()


def resnet_50_final_1():
    def bottle_neck(output, n_filters, stride):
        output = keras.layers.Conv2D(n_filters, [1, 1], stride, 'same')(output)
        output = keras.layers.Conv2D(n_filters, [3, 3], 1, 'same')(output)
        output = keras.layers.Conv2D(n_filters * 4, [1, 1], 1, 'same')(output)
        return output

    def resnet_block(output, n_filters, stride, n):

        shortcut = keras.layers.Conv2D(n_filters * 4, [1, 1], stride, 'same')(output)


        for _ in range(n):
            output = bottle_neck(output, n_filters, stride)

            output = keras.layers.add([output, shortcut])
            shortcut = output
            stride = 1

        return shortcut

    inputs = keras.layers.Input(shape=[224, 224, 3])
    output = inputs

    # conv 1
    output = keras.layers.Conv2D(64, [7, 7], 2, 'same')(output)
    output = keras.layers.MaxPool2D([2, 2], 2)(output)

    # conv 2

    output = resnet_block(output, 64, 1, 3)

    # conv 3

    output = resnet_block(output, 128, 2, 4)

    # conv 4

    output = resnet_block(output, 256, 2, 6)

    # conv 5

    output = resnet_block(output, 512, 2,3)

    # ----------------------
    output = keras.layers.GlobalAveragePooling2D()(output)
    output = keras.layers.Dense(1000, activation='softmax')(output)

    model = keras.models.Model(inputs, output)
    model.summary()


def resnet_50_final_2():
    def bottle_neck(output, n_filters, stride):
        output = keras.layers.Conv2D(n_filters, [1, 1], stride, 'same')(output)
        output = keras.layers.BatchNormalization()(output)
        output = keras.layers.ReLU()(output)


        output = keras.layers.Conv2D(n_filters, [3, 3], 1, 'same')(output)
        output = keras.layers.BatchNormalization()(output)
        output = keras.layers.ReLU()(output)

        output = keras.layers.Conv2D(n_filters * 4, [1, 1], 1, 'same')(output)
        output = keras.layers.BatchNormalization()(output)

        return output

    def resnet_block(output, n_filters, stride, n):

        shortcut = keras.layers.Conv2D(n_filters * 4, [1, 1], stride, 'same')(output)


        for _ in range(n):
            output = bottle_neck(output, n_filters, stride)

            output = keras.layers.add([output, shortcut])
            output = keras.layers.ReLU()(output)
            shortcut = output
            stride = 1

        return shortcut

    inputs = keras.layers.Input(shape=[224, 224, 3])
    output = inputs

    # conv 1
    output = keras.layers.Conv2D(64, [7, 7], 2, 'same')(output)
    output = keras.layers.MaxPool2D([2, 2], 2)(output)

    # conv 2

    output = resnet_block(output, 64, 1, 3)

    # conv 3

    output = resnet_block(output, 128, 2, 4)

    # conv 4

    output = resnet_block(output, 256, 2, 6)

    # conv 5

    output = resnet_block(output, 512, 2,3)

    # ----------------------
    output = keras.layers.GlobalAveragePooling2D()(output)
    output = keras.layers.Dense(1000, activation='softmax')(output)

    model = keras.models.Model(inputs, output)
    model.summary()

resnet_50_final_2()