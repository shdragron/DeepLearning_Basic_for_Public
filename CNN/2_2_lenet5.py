# 2_2_lenet5.py


import keras
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np


mnist = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist

# 퀴즈: x_train 데이터에서 앞쪽 5개를 그래프에 출력하세요.

def image_show(images,size):
    plt.figure(figsize=(2*size, 5))
    for i in range(size):
        plt.subplot(1, size, i+1)
        plt.imshow(images[i])
    plt.tight_layout()
    plt.show()



x_train, x_test = x_train.reshape(-1,28,28,1), x_test.reshape(-1,28,28,1) # 데이터 개수, feature (이미지의 각점(픽셀) 하나가 feature이다)

x_train, x_test = x_train / 255.0, x_test / 255.0 # 정규화

# image_show(x_train,10)



model = keras.models.Sequential([

    keras.layers.Input(shape = x_train[0].shape),
    # convolution Layer
    # 156 = 1 * 5*5 * 6 + 6
    keras.layers.Conv2D(6,5,1,'same',activation='relu',name='conv_1'),
    keras.layers.MaxPooling2D(2,name='pool_1'),
    # 2416 = 6 * 5*5 * 16 + 16
    keras.layers.Conv2D(16, 5, 1, 'valid', activation='relu', name='conv_2'),
    keras.layers.MaxPooling2D(2, name='pool_2'),


    keras.layers.Flatten(name='flatten'),
    keras.layers.Dense(units=120, activation='relu',name='C_5'),
    keras.layers.Dense(units=84, activation='relu',name='F_6'),
    keras.layers.Dense(units=10, activation='softmax',name='Output'),

])

model.summary() # 모델의 아키텍쳐를 보여준다.


model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=0,verbose = 1,
          validation_data=(x_test,y_test),
          batch_size= 100)


p = model.predict(x_test)
# print(p)

print(p.shape)


p_max = np.argmax(p,axis=1)
print(np.argmax(p,axis=1).shape)


image_show(y_right,10)

equals = (p_max == y_test)

x_right = x_test[equals]
x_wrong = x_test[~equals]
y_right = y_test[equals]
y_wrong = y_test[~equals]
print(x_right.shape)
print(y_right.shape)
print(x_wrong.shape)
print(y_wrong.shape)
# 퀴즈: 예측한 결과로 부터 틀린거 10개를 그래프로 표시하세요.
