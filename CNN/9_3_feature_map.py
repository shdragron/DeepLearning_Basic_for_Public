# 9_3_feature_map.py
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

model_path = './models/dogs_and_cats.keras'
model_origin = keras.models.load_model(model_path)

layers = model_origin.layers
conv_base = layers[0]
print(layers)
print(conv_base)
print(conv_base.layers)
print(conv_base.layers[0])
print(conv_base.layers[1])
print(conv_base.layers[0].output)
print(conv_base.layers[1].output)   # (None, 150, 150, 64)

model = keras.Model(layers[0].input, conv_base.layers[1].output)
model.summary()

img_path = 'dosg-vs-cats/dog.805.jpg'
img = keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
img_array = keras.preprocessing.image.img_to_array(img)
print(type(img))
print(type(img_array))
print(img_array.shape)

img_array = img_array[np.newaxis]

feature_maps = model.predict(img_array, verbose=0)
print(feature_maps.shape)       # (1, 150, 150, 64)

# 퀴즈
# 앞쪽 8장의 피처맵에 대해서만 그래프로 그려주세요
for i in range(9):
    plt.subplot(3, 3, i+1)

    plt.imshow(feature_maps[0, :, :, i])
    plt.axis('off')

plt.show()

