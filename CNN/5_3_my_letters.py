# 5_3_my_letters.py

# 퀴즈:
# 우리가 만든 숫자 데이터에 대해 딥러닝 모델을 구축하세요. (제너레이터 사용)

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

INPUT_SIZE = 56

gen_train = ImageDataGenerator(rescale=1 - 1./255,
                               rotation_range=0.25,
                               brightness_range=[4,5],)

flow_train = gen_train.flow_from_directory('letters/digit/train',
                               batch_size=4,
                               target_size=(INPUT_SIZE, INPUT_SIZE),
                               class_mode='sparse',)

gen_test = ImageDataGenerator(rescale=1 - 1./255,)

flow_test = gen_test.flow_from_directory('letters/digit/test',
                               batch_size=4,
                               target_size=(INPUT_SIZE, INPUT_SIZE),
                               class_mode='sparse',)



# model = keras.models.Sequential([
#
#     keras.layers.Input(shape = [30,30,3]),
#     # convolution Layer
#     # 156 = 1 * 5*5 * 6 + 6
#     keras.layers.Conv2D(6,5,1,'same',activation='relu',name='conv_1'),
#     keras.layers.Conv2D(6,5,2,'same',activation='relu',name='conv_2'),
#     # 2416 = 6 * 5*5 * 16 + 16
#     keras.layers.Conv2D(16, 5, 1, 'valid', activation='relu', name='conv_3'),
#     keras.layers.MaxPooling2D(2, name='pool_2'),
#
#     # FC Layer
#     # keras.layers.Flatten(),
#     keras.layers.Flatten(name='flatten'),
#     # 48120 = 400 * 120 + 120
#     keras.layers.Dense(units=120, activation='relu',name='C_5'),
#     # 10164 = 120 * 84 + 84
#     keras.layers.Dense(units=84, activation='relu',name='F_6'),
#     #  Output     Input      weight + bias
#     # (? , 10) = (? , 84) @ (84, 10) + 10
#     keras.layers.Dense(units=10, activation='softmax',name='Output'),
# ])

model = keras.Sequential([
    keras.layers.Input(shape=[INPUT_SIZE, INPUT_SIZE, 3]),
    keras.layers.Conv2D(16, [3, 3], 1, 'same',),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(16, [3, 3], 1, 'same', ),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D([2, 2], 2),

    keras.layers.Conv2D(32, [3, 3], 1, 'same', ),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.Conv2D(32, [3, 3], 1, 'same', ),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D([2, 2], 2),

    keras.layers.Conv2D(64, [3, 3], 1, 'same', ),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(64, [3, 3], 1, 'same',),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D([2, 2], 2),



    # top
    keras.layers.Conv2D(256, [7, 7], 1, 'valid', activation='relu'),
    keras.layers.Conv2D(256, [1, 1], 1, activation='relu'),
    keras.layers.Conv2D(10, [1, 1], 1, activation='softmax'),
    keras.layers.Flatten(),
])

model.compile(optimizer=keras.optimizers.Adam(0.0001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

# model.fit(flow_train, epochs=10, verbose=1)

model.fit(flow_train, epochs=100, verbose=1,validation_data=flow_test)




