# 4_5_generator.py

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 퀴즈 test data에 대해 generator를 연결하고 결과를 확인하세요.

gen_train = ImageDataGenerator(rescale=1./255,
                               zoom_range=10,
                               rotation_range=0.45,)

flow_train = gen_train.flow_from_directory('jpg/flowers5/train',
                               batch_size=32,
                               target_size=(224, 224),
                               class_mode='sparse',)

gen_test = ImageDataGenerator(rescale=1./255)

flow_test = gen_test.flow_from_directory('jpg/flowers5/test',
                               batch_size=32,
                               target_size=(224, 224),
                               class_mode='sparse',)



model = keras.Sequential([
    keras.layers.Input(shape=[224, 224, 3]),
    keras.layers.Conv2D(32, [11, 11], 4, 'valid', activation='relu'),
    keras.layers.MaxPool2D([3, 3], 2),

    keras.layers.Conv2D(96, [5, 5], 1, 'same', activation='relu'),
    keras.layers.MaxPool2D([3, 3], 2),

    keras.layers.Conv2D(192, [3, 3], 1, 'same', activation='relu'),
    keras.layers.Conv2D(192, [3, 3], 1, 'same', activation='relu'),
    keras.layers.Conv2D(128, [3, 3], 1, 'same', activation='relu'),
    keras.layers.MaxPool2D([3, 3], 2),

    # top
    keras.layers.Conv2D(256, [5, 5], 1, 'valid', activation='relu'),
    keras.layers.Conv2D(256, [1, 1], 1, activation='relu'),
    keras.layers.Conv2D(5, [1, 1], 1, activation='softmax'),
    keras.layers.Flatten(),


])

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

# model.fit(flow_train, epochs=10, verbose=1)

model.fit(flow_train, epochs=100, verbose=1,validation_data=flow_test)

