# 7_6_pretrained.py
import keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def extract_features(conv_base, gen, path, n_samples, batch_size, target_size):
    x = np.zeros([n_samples, 2, 2, 512])
    y = np.zeros([n_samples])

    flow = gen.flow_from_directory(path,
                                   target_size=target_size,
                                   batch_size=batch_size,
                                   class_mode='sparse')

    for i, (xx, yy) in enumerate(flow):
        n1 = i * batch_size
        n2 = n1 + batch_size

        if n2 > n_samples:
            remainder = n_samples - n1
            x[n1:] = conv_base.predict(xx[:remainder])
            y[n1:] = yy[:remainder]
            break

        x[n1:n2] = conv_base.predict(xx)
        y[n1:n2] = yy

    return x.reshape(-1, 2 * 2 * 512), y


gen = ImageDataGenerator(rescale=1/255)

conv_base = keras.applications.VGG16(include_top=False, input_shape=(64, 64, 3))
conv_base.trainable = False
conv_base.summary()

x_train, y_train = extract_features(conv_base, gen, 'flowers5/train', 200, 32, [64, 64])
x_test, y_test = extract_features(conv_base, gen, 'flowers5/test', 100, 32, [64, 64])

model = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(5, activation='softmax'),
])
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=100, verbose=2, validation_data=(x_test, y_test))



