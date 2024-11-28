# 6_1_auto_encoder_basic.py

import keras
import matplotlib.pyplot as plt

def get_data():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)



    return x_train, x_test



def show_digit_double(x_test, predictions):

    plt.figure(figsize=(15,3))
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(x_test[i].reshape(28,28))
        plt.axis('off')
        plt.gray()

        plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(predictions[i].reshape(28, 28))
        plt.axis('off')
        plt.gray()

    plt.tight_layout()
    plt.show()

def make_single():
    return keras.Sequential([
        keras.layers.Input(shape=[784]),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(784, activation='sigmoid'),
    ])

def make_multi():
    return keras.Sequential([
        keras.layers.Input(shape=[784]),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(784, activation='sigmoid'),
    ])

def make_conv():
    return keras.Sequential([
        keras.layers.Input(shape=[28, 28, 1]),
        keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu'),
        keras.layers.MaxPool2D([2, 2], 2),
        keras.layers.Conv2D(8, [3, 3], 1, 'same', activation='relu'),
        keras.layers.MaxPool2D([2, 2], 2),

        keras.layers.Conv2D(8, [3, 3], 1, 'same', activation='relu'),
        keras.layers.UpSampling2D(size=(2, 2)),
        keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu'),
        keras.layers.UpSampling2D(size=(2, 2)),
        keras.layers.Conv2D(1, [3, 3], 1, 'same', activation='sigmoid'),
    ])



x_train, x_test = get_data()

model = make_conv()

model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'],
              loss=keras.losses.binary_crossentropy)

# 컨볼루션 레이어 전용
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model.fit(x_train, x_train, verbose=1, epochs=10, batch_size=100)

p = model.predict(x_test, verbose=0)
show_digit_double(x_test, p)
