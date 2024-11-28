# 5_3_rock_paper_scissors.py
import keras
import tensorflow as tf


# 텐서플로 cnn 문제
# 케라스에 있는 이미지 제너레이터를 사용해서 cnn 모델을 구축하세요
# 이미지 크기는 (150, 150)으로 사용합니다
# 통과 기준 : 80%
def make_sequential():
    conv_base = keras.applications.VGG16(include_top=False, input_shape=(150, 150, 3))
    conv_base.trainable = False

    model = keras.Sequential([
        conv_base,
        # keras.layers.GlobalAveragePooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.summary()

    return model


gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                            zoom_range=0.1,
                                                            rotation_range=20,
                                                            horizontal_flip=True)
flow_train = gen_train.flow_from_directory(
    'Rock-Paper-Scissors/train',
    target_size=(150, 150),
    class_mode='sparse',
    batch_size=32
)

gen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
flow_test = gen_test.flow_from_directory(
    'Rock-Paper-Scissors/test',
    target_size=(150, 150),
    class_mode='sparse',
    batch_size=32
)

model = make_sequential()

model.compile(optimizer=keras.optimizers.RMSprop(0.0001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(flow_train, epochs=10, verbose=1, validation_data=flow_test)
