import keras
from sklearn import preprocessing

flowers = __import__('3_2_flowers')

x_train, y_train = flowers.load_images('jpg/flowers5/train/Buttercup', 0)
x_test, y_test = flowers.load_images('jpg/flowers5/test/Buttercup', 1)

x_train, y_train = x_train / 255, y_train / 255
x_test, y_test = x_test / 255, y_test / 255
print(x_train.shape, y_train.shape)

# layer 1
input1 = keras.layers.Input(shape=(x_train[0].shape))
output_1_1 = keras.layers.Conv2D(48, (11, 11), strides=4, padding='VALID', activation='relu')(input1)
output_1_1 = keras.layers.MaxPool2D((5, 5), strides=2, padding='same')(output_1_1)

output_1_2 = keras.layers.Conv2D(48, (11, 11), strides=4, padding='VALID', activation='relu')(input1)
output_1_2 = keras.layers.MaxPool2D((5, 5), strides=2, padding='same')(output_1_2)

# layer 2
output_2_1 = keras.layers.Conv2D(128, (5, 5), strides=2, padding='VALID', activation='relu')(output_1_1)
output_2_1 = keras.layers.MaxPool2D((3, 3), strides=2, padding='same')(output_2_1)

output_2_2 = keras.layers.Conv2D(128, (5, 5), strides=2, padding='VALID', activation='relu')(output_1_2)
output_2_2 = keras.layers.MaxPool2D((3, 3), strides=2, padding='same')(output_2_2)

# layer 3
output_3_1 = keras.layers.Conv2D(192, (3, 3), strides=2, padding='same', activation='relu')(output_2_1)
output_3_2 = keras.layers.Conv2D(192, (3, 3), strides=2, padding='same', activation='relu')(output_2_2)
output_3_3 = keras.layers.Conv2D(192, (3, 3), strides=2, padding='same', activation='relu')(output_2_1)
output_3_4 = keras.layers.Conv2D(192, (3, 3), strides=2, padding='same', activation='relu')(output_2_2)

# layer 4
output_4_1 = keras.layers.concatenate([output_3_1, output_3_2])
output_4_2 = keras.layers.concatenate([output_3_3, output_3_4])

output_4_3 = keras.layers.Conv2D(192, (3, 3), strides=1, padding='same', activation='relu')(output_4_2)
output_4_4 = keras.layers.Conv2D(192, (3, 3), strides=1, padding='same', activation='relu')(output_4_3)

# layer 5
output_5_1 = keras.layers.Conv2D(128, (3, 3), strides=1, padding='VALID', activation='relu')(output_4_3)
output_5_2 = keras.layers.Conv2D(128, (3, 3), strides=1, padding='VALID', activation='relu')(output_4_4)

output_5_3 = keras.layers.MaxPool2D((3, 3), strides=1, padding='VALID')(output_5_1)
output_5_4 = keras.layers.MaxPool2D((3, 3), strides=1, padding='VALID')(output_5_2)

output_5_6 = keras.layers.Flatten()(output_5_3)
output_5_7 = keras.layers.Flatten()(output_5_4)

# FC
output_6_1 = keras.layers.Dense(2048, activation='relu')(output_5_6)
output_6_2 = keras.layers.Dense(2048, activation='relu')(output_5_7)

output_6_5 = keras.layers.concatenate([output_6_1, output_6_2])

# FC_2
output_7_1 = keras.layers.Dense(2048, activation='relu')(output_6_5)

# FC_3
output_8_3 = keras.layers.Dense(5, activation='softmax')(output_7_1)

model = keras.models.Model(input1, outputs=output_8_3)

enc = preprocessing.LabelEncoder()

y_train = enc.fit_transform(y_train)
y_test = enc.fit_transform(y_test)

model.compile(optimizer=keras.optimizers.RMSprop(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=2, verbose=2)

model.summary()
