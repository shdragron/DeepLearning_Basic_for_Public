# 6_3_auto_encoder.py
from keras import layers, models, datasets, callbacks
import numpy as np
import matplotlib.pyplot as plt
import keras

IMAGE_SIZE = 32
LATENT_DIM = 2


def show_digit_double(x_test, predictions):
    plt.figure(figsize=[15, 3])
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.imshow(x_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE))
        plt.gray()
        plt.axis('off')

        plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(predictions[i].reshape(IMAGE_SIZE, IMAGE_SIZE))
        plt.gray()
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# 레이블 설명
# 0 티셔츠/탑
# 1 바지
# 2 풀오버(스웨터의 일종)
# 3 드레스
# 4 코트
# 5 샌들
# 6 셔츠
# 7 스니커즈
# 8 가방
# 9 앵클 부츠
def load_fashion_mnist():
    data = datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data
    return x_train, x_test, y_train, y_test


def preprocess(imgs):
    imgs = imgs / 255
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)))
    imgs = np.expand_dims(imgs, -1)
    return imgs


def make_auto_encoder():
    # 인코더
    enc_in = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = layers.Conv2D(32, (3, 3), 2, 'same', activation='relu')(enc_in)
    x = layers.Conv2D(64, (3, 3), 2, 'same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), 2, 'same', activation='relu')(x)
    shape_before_flattening = x.shape[1:]

    x = layers.Flatten()(x)
    enc_out = layers.Dense(LATENT_DIM)(x)        # 시각화에 사용하기 위해 2차원 적용

    encoder = models.Model(enc_in, enc_out)
    # encoder.summary()

    # 디코더
    dec_in = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(np.prod(shape_before_flattening))(dec_in)
    x = layers.Reshape(shape_before_flattening)(x)
    x = layers.Conv2DTranspose(128, (3, 3), 2, 'same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (3, 3), 2, 'same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, (3, 3), 2, 'same', activation='relu')(x)
    dec_out = layers.Conv2D(1, (3, 3), 1, 'same', activation='sigmoid')(x)

    decoder = models.Model(dec_in, dec_out)
    # decoder.summary()

    auto_encoder = models.Model(enc_in, decoder(enc_out))
    # auto_encoder.summary()

    return encoder, decoder, auto_encoder

if __name__ == '__main__': # 6_5 에서 import하기 위해

    x_train, x_test, y_train, y_test = load_fashion_mnist()
    # show_digit_double(x_train, x_test)

    # print(x_train.shape)              # (60000, 28, 28)

    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    # print(x_train.shape)              # (60000, 32, 32, 1)

    encoder, decoder, auto_encoder = make_auto_encoder()

    auto_encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    checkpoints = keras.callbacks.ModelCheckpoint('./models/ae--{epoch:02d}-{val_acc:.2f}.weights.h5',
                                                  monitor='val_loss',
                                                  verbose=1,
                                                  save_best_only=True,
                                                  save_weights_only=True,)

    history = auto_encoder.fit(x_train, x_train, epochs=100, batch_size=100,
                               shuffle=True, validation_data=(x_test, x_test),
                               callbacks=[checkpoints],)


    p = auto_encoder.predict(x_test, verbose=0)
    show_digit_double(x_test, p)



