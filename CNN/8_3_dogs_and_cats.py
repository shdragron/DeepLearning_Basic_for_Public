# 8_3_dogs_and_cats.py

import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
# 퀴즈: 개와 고양이 데이터셋에 대해 70%로 학습하고 30%에 대해 결과를 구하세요.
# 네트윅 2개(입력크기1 244, 입력크기2 192)


def make_folder():
    for ch in ['dog','cat']:
        os.makedirs(os.path.join('dogs_cats/test/{}'.format(ch)),exist_ok=True)
        os.makedirs(os.path.join('dogs_cats/train/{}'.format(ch)),exist_ok=True)

def copy_image():
    TRAIN_SIZE = 1400
    for image in os.listdir('dosg-vs-cats'):  # 오타 수정: 'dosg-vs-cats' -> 'dogs-vs-cats'
        animal, num, ext = image.split('.')
        print(animal, num, ext)
        target = 'train' if int(num) < TRAIN_SIZE else 'test'
        dst_dir = os.path.join('dogs_cats/{}/{}/'.format(target, animal))
        dst = os.path.join(dst_dir, image)  # 파일 이름을 포함한 경로로 수정
        src = os.path.join('dosg-vs-cats/', image)
        shutil.copyfile(src, dst)


def data_split():
    INPUT_SIZE = 224

    gen_train = ImageDataGenerator(rescale=1/ 255,
                                   rotation_range=0.25,
                                   brightness_range=[4, 5], )

    flow_train = gen_train.flow_from_directory('dogs_cats/train',
                                               batch_size=4,
                                               target_size=(INPUT_SIZE, INPUT_SIZE),
                                               class_mode='binary', )

    gen_test = ImageDataGenerator(rescale=1/255, )

    flow_test = gen_test.flow_from_directory('dogs_cats/test',
                                             batch_size=4,
                                             target_size=(INPUT_SIZE, INPUT_SIZE),
                                             class_mode='binary', )
    return flow_train, flow_test

# make_folder()

# copy_image()

train, test= data_split()

vgg_base = keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3))
vgg_base.trainable = False
# vgg_base.summary()

res_base = keras.applications.ResNet50(include_top=False, input_shape=(192, 192, 3))
res_base.trainable = False
# res_base.summary()

input1 = keras.layers.Input(shape=(224, 224, 3))
output1 = vgg_base(input1) #  (None, 7, 7, 512)
output1 = keras.layers.Flatten()(output1) # (None, 25088)



input2 = keras.layers.Input(shape=(192, 192, 3))
output2 = res_base(input2) #  (None, 7, 7, 512)
output2 = keras.layers.Flatten()(output2) # (None, 25088)


output = keras.layers.concatenate([output1, output2])
output = keras.layers.Dense(1, activation='sigmoid')(output)

model = keras.models.Model(inputs=[input1,input2], outputs=output)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.binary_crossentropy,
              metrics=['acc'])

model.fit(train,
          epochs=100,
          verbose=1,
          validation_data=test,)
