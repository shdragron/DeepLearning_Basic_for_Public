import keras
import numpy as np
import matplotlib.pyplot as plt


def make_generator():
    gen_dense_size = (7, 7, 64)

    inputs = keras.layers.Input((100,))     # make_noise에 전달된 latent_dim과 동일

    x = keras.layers.Dense(units=np.prod(gen_dense_size))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Reshape(gen_dense_size)(x)

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=5, padding='same', strides=1)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', strides=1)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', strides=1)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=1, kernel_size=5, padding='same', strides=1)(x)
    outputs = keras.layers.Activation('sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.summary()

    return model


def make_discriminator():
    inputs = keras.layers.Input((28, 28, 1))

    x = keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(inputs)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(rate=0.4)(x)

    x = keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(rate=0.4)(x)

    x = keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(rate=0.4)(x)

    x = keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same')(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(rate=0.4)(x)

    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='he_normal')(x)

    model = keras.Model(inputs, outputs)
    model.summary()

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0008),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc'])
    return model


def make_adversarial(generator, discriminator):
    inputs = keras.layers.Input((100,))
    outputs = generator(inputs)
    outputs = discriminator(outputs)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0004),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc'])
    return model


def make_noise(batch_size, latent_dim):
    return np.random.normal(0, 1, [batch_size, latent_dim])


def make_mixed_images(images, generator, batch_size):
    # idx = np.random.choice(range(len(images)), batch_size, replace=False)
    idx = np.random.randint(0, len(images), batch_size)
    real_images = images[idx]

    noises = make_noise(batch_size, latent_dim=100)
    fake_images = generator.predict(noises, verbose=0)

    bind_images = np.concatenate([real_images, fake_images], axis=0)
    bind_labels = np.concatenate([np.ones([batch_size, 1]), np.zeros([batch_size, 1])])      # 진짜 + 가짜

    return bind_images, bind_labels


def train_gan(batch_size):
    generator = make_generator()
    discriminator = make_discriminator()
    adversarial = make_adversarial(generator, discriminator)

    camel = np.load('data/full_numpy_bitmap_camel.npy')     # (121399, 784)
    camel = np.reshape(camel, (-1, 28, 28, 1))
    camel = camel / 255

    for epoch in range(6):
        for i in range(1000):
            # 판별자 모델 학습 (가짜와 진짜가 섞여 있다)
            images, labels = make_mixed_images(camel, generator, batch_size)

            discriminator.trainable = True
            discriminator.train_on_batch(images, labels)

            # --------------------------------------------- #
            # 적대적 모델 학습
            discriminator.trainable = False

            noises = make_noise(batch_size, latent_dim=100)
            target = np.ones([batch_size, 1])
            adversarial.train_on_batch(noises, target)

            if i % 10 == 9:
                print(i+1, '---')
                generator.save_weights('models/camel_generator_{}_{:04}.weights.h5'.format(epoch, i+1))

        print('epoch :', epoch)


def show_generator():
    generator = make_generator()
    # discriminator = make_discriminator()
    # adversarial = make_adversarial(generator, discriminator)

    # adversarial.load_weights('models/camel_adversarial.weights.h5')
    # discriminator.load_weights('models/camel_discriminator.weights.h5')
    generator.load_weights('models/camel_generator_3_0110.weights.h5')

    samples = make_noise(batch_size=10, latent_dim=100)
    # p = samples.reshape(-1, 10, 10)
    p = generator.predict(samples, verbose=0)
    # print(p.shape)

    plt.figure(figsize=(15, 2))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(p[i], cmap='Greys')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# train_gan(batch_size=64)
show_generator()
