# 7_5_pretrained.py
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gen_train = ImageDataGenerator(rescale=1/255)
flow_train = gen_train.flow_from_directory(
    'flowers5/train',
    target_size=(224, 224),
    class_mode='sparse',
    batch_size=32
)
gen_test = ImageDataGenerator(rescale=1/255)
flow_test = gen_test.flow_from_directory(
    'flowers5/test',
    target_size=(224, 224),
    class_mode='sparse',
    batch_size=32
)

# conv_base = keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3))
conv_base = keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3))
conv_base.trainable = False

model = keras.Sequential([
    conv_base,

    # top
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(5, activation='softmax'),
])
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(flow_train, epochs=100, verbose=2, validation_data=flow_test)
