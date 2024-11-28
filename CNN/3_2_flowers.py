# 3_2_flowers.py

import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
# 퀴즈: 현재 폴도의 파일 이름을 모두 출력해주세요
def show_samples():
    files = os.listdir('.')
    print(files)
    for name in files:
        if os.path.isfile(name):
            print(name)

    # 아래 구조 처럼 폴더를 만드세요.

    # root +---- first  +---- one
    #                   +---- two
    #      +---- second +---- one
    #                   +---- two

    os.mkdir('root')
    os.chdir('./root')
    os.mkdir('first')
    os.chdir('./first')
    os.mkdir('one')
    os.mkdir('two')
    os.chdir('..')
    os.mkdir('second')
    os.chdir('./second')
    os.mkdir('one')
    os.mkdir('two')

# flowers5 +- train +- Buttercup
#                   +- ColtsFoot
#                   +- Daffodil
#                   +- Daisy
#                   +- Dandelion
#          +- test  +- Buttercup
#                   +- ColtsFoot
#                   +- Daffodil
#                   +- Daisy
#                   +- Dandelion
def make_flowers_folder():

    os.chdir('jpg')
    os.mkdir('flowers5')
    os.chdir('flowers5')
    os.mkdir('train')
    os.chdir('train')
    for i in ['Buttercup','ColtsFoot','Daffodil','Daisy','Dandelion']:
        os.mkdir(i)
        # os.mkdirs(os.path.join('flowers5/test', i))
        # os.mkdirs(os.path.join('flowers5/train', i))

    os.chdir('..')
    os.mkdir('test')
    os.chdir('test')
    for i in ['Buttercup','ColtsFoot','Daffodil','Daisy','Dandelion']:
        os.mkdir(i)

# make_flowers_folder()
#  퀴즈 첫번째 꽃파일 하나
#  를 train/Buttercup으로 복사하세요.
print(os.listdir('.'))
def data_seperator():
    os.chdir('jpg')
    j = 0
    for i in range(400):
        i = i+1
        j += 1
        if i < 81:
            if j <= 60:
                if i < 10:
                    shutil.copy(('image_000'+ str(i) + '.jpg'),'flowers5/train/Buttercup')
                elif i < 100:
                    shutil.copy(('image_00'+ str(i) + '.jpg'),'flowers5/train/Buttercup')
                else:
                    shutil.copy(('image_0'+ str(i) + '.jpg'),'flowers5/train/Buttercup')
            elif j > 60:
                if i < 10:
                    shutil.copy(('image_000'+ str(i) + '.jpg'),'flowers5/test/Buttercup')
                elif i < 100:
                    shutil.copy(('image_00'+ str(i) + '.jpg'),'flowers5/test/Buttercup')
                elif i >= 100:
                    shutil.copy(('image_0'+ str(i) + '.jpg'),'flowers5/test/Buttercup')

        elif i < 161:
            if j <= 140:
                if i < 100:
                    shutil.copy(('image_00' + str(i) + '.jpg'), 'flowers5/train/ColtsFoot')
                else:
                    shutil.copy(('image_0' + str(i) + '.jpg'), 'flowers5/train/ColtsFoot')
            elif j > 140:
                shutil.copy(('image_0'+ str(i) + '.jpg'),'flowers5/test/ColtsFoot')
        elif i < 241:
            if j <= 220:
                shutil.copy(('image_0'+ str(i) + '.jpg'),'flowers5/train/Daffodil')
            elif j > 220:
                shutil.copy(('image_0'+ str(i) + '.jpg'),'flowers5/test/Daffodil')
        elif i < 321:
            if j <= 300:
                shutil.copy(('image_0'+ str(i) + '.jpg'),'flowers5/train/Daisy')
            elif j > 300:
                shutil.copy(('image_0' + str(i) + '.jpg'), 'flowers5/test/Daisy')
        elif i < 401:
            if j <= 380:
                shutil.copy(('image_0'+ str(i) + '.jpg'),'flowers5/train/Dandelion')
            elif j > 380:
                shutil.copy(('image_0'+ str(i) + '.jpg'),'flowers5/test/Dandelion')


def data_seperator_teach():

    for idx, flower in enumerate(['Buttercup','ColtsFoot','Daffodil','Daisy','Dandelion']):
        base = idx * 80
        for i in range(60):
            src = 'jpg/image_{:04}.jpg'.format(i+1)
            dst = 'flowers5/train/{}/image_{:04}.jpg'.format(flower,base+i+1)
            shutil.copy(src, dst)

    for idx, flower in enumerate(['Buttercup','ColtsFoot','Daffodil','Daisy','Dandelion']):
        base = idx * 80
        for i in range(60, 80):
            src = 'jpg/image_{:04}.jpg'.format(i+1)
            dst = 'flowers5/test/{}/image_{:04}.jpg'.format(flower,base+i+1)
            shutil.copy(src, dst)

# 퀴즈: 전달 받은 폴더에 포함된 이미지를 읽어서 넘파이 배열로 변환하세요.
def load_images(path,kinds):
    x_train,x_test,y = [], [], []
    if kinds == 0:
        for idx, flower in enumerate(['Buttercup','ColtsFoot','Daffodil','Daisy','Dandelion']):
            base = idx * 80
            for i in range(60):
                img = Image.open('flowers5/train/{}/image_{:04}.jpg'.format(flower, base + i+1))
                img_resized = img.resize([224,224])
                img_resized = np.array(img_resized)
                x_train.append(img_resized)
                y.append(idx)
        result_x = np.array(x_train)
    else:
        for idx, flower in enumerate(['Buttercup', 'ColtsFoot', 'Daffodil', 'Daisy', 'Dandelion']):
            base = idx * 80
            for i in range(60,80):
                img = Image.open('flowers5/test/{}/image_{:04}.jpg'.format(flower,base+i+1))
                img_resized = img.resize([224, 224])
                img_resized = np.array(img_resized)
                x_test.append(img_resized)
                y.append(idx)
        result_x = np.array(x_test)
    return result_x, np.array(y)

# make_flowers_folder()
data_seperator()
x_train, y_train = load_images('jpg/flowers5/train/Buttercup',0)
x_test, y_test = load_images('jpg/flowers5/test/Buttercup',1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)