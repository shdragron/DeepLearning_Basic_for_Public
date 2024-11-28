# 4_4_generator.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

gen = ImageDataGenerator(rescale=1./255)

flow = gen.flow_from_directory('jpg/flowers5/train',
                               batch_size=4,
                               target_size=(224, 224),
                               class_mode='sparse',)
images = next(flow)
# print(images)
print(type(images),len(images))

x, y = images
print(x.shape,y.shape)
print(y)

plt.figure(num = 1, figsize = [8, 2])
for i in range(4):
    ax = plt.subplot(1, 4, i+1)
    ax.axis('off')
    plt.imshow(x[i])

plt.tight_layout()
plt.show()