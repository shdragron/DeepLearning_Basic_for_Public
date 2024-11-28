# 7_2_vae.py
from keras import layers, models, datasets, callbacks, metrics, losses
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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


def load_fashion_mnist():
    data = datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data
    return x_train, x_test, y_train, y_test


def preprocess(imgs):
    imgs = imgs / 255
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)))
    imgs = np.expand_dims(imgs, -1)
    return imgs


class Sampling(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        # epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def make_vae():
    # 인코더
    enc_in = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = layers.Conv2D(32, (3, 3), 2, 'same', activation='relu')(enc_in)
    x = layers.Conv2D(64, (3, 3), 2, 'same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), 2, 'same', activation='relu')(x)
    shape_before_flattening = x.shape[1:]

    x = layers.Flatten()(x)

    # enc_out = layers.Dense(LATENT_DIM)(x)        # 오토인코더 코드
    # encoder = models.Model(enc_in, enc_out)
    # 이미지(원본) -> 숫자 2개 -> 이미지(생성) : 오토인코더
    # 이미지(원본) -> 평균/분산 -> 이미지(생성) : 가변형 오토인코더
    z_mean = layers.Dense(LATENT_DIM)(x)
    z_log_var = layers.Dense(LATENT_DIM)(x)
    z = Sampling()(z_mean, z_log_var)

    encoder = models.Model(enc_in, [z_mean, z_log_var, z])
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

    vae = VAE(encoder, decoder)
    # vae.summary()

    return encoder, decoder, vae


class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        # 재구성 오차 - 원본과 예측 사이의 오차
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        # 잠재 손실(쿨백-라이블러 발산) - 분포가 얼마나 다른지 측정
        # 평균과 분산이 0일 때 최소
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # call 함수를 호출하기 때문에 반환값이 3개
            z_mean, z_log_var, reconstruction = self(data)
            # 원본 (10, 32, 32, 1)
            # 생성 (10, 32, 32, 1)
            reconstruction_loss = tf.reduce_mean(
                500 * losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3))
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # y는 필요없기 때문에 버린다
        # x, y = data
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, reconstruction = self(data)
        reconstruction_loss = tf.reduce_mean(
            500 * losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3))
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
        )
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_fashion_mnist()

    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    encoder, decoder, vae = make_vae()

    checkpoints = callbacks.ModelCheckpoint('./models/vae-{epoch:02d}-{val_loss:.2f}.weights.h5',
                                            monitor='val_loss', verbose=1,
                                            save_best_only=True, save_weights_only=True)

    vae.compile(optimizer='adam', loss='binary_crossentropy')
    vae.fit(x_train, epochs=100, batch_size=100,
            validation_data=x_test, callbacks=[checkpoints])

    z_mean, z_log_var, reconstruction = vae.predict(x_test, verbose=0)
    show_digit_double(x_test, reconstruction)
