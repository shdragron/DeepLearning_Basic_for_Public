# 1_1_cnn_basic.py

import keras
from sklearn import preprocessing, model_selection

mnist = keras.datasets.mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist
x_train, x_test = x_train.reshape(-1,28,28,1), x_test.reshape(-1,28,28,1) # 데이터 개수, feature (이미지의 각점(픽셀) 하나가 feature이다)

x_train, x_test = x_train / 255.0, x_test / 255.0 # 정규화

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(y_train[:10])
print(type(y_train[0]))

model = keras.models.Sequential([

    keras.layers.Input(shape = x_train[0].shape),
    # convolution Layer

    # Con2D는 2차원으로 슬라이딩 진행한다.
    keras.layers.Conv2D(
        filters=6,
        kernel_size=(5,5),
        strides=(1,1),
        padding='VALID', # VALID: 끝에 도달하여 튀어나오기전까지만 진행 -> padding 진행, SAME: 끝에  도달해도 마지막까지 먹을 수 있게 padding 진행
        # 스트라이드 1에 패딩: 'SAME'이면 원본과 같은 크기로 만든다.
        activation='relu',
    ),
    keras.layers.MaxPooling2D(pool_size=(2,2),
                              strides=(2,2),
                              padding='valid',),

    # keras.layers.Conv2D(6,5,1,'VALID',activation='relu'),
    # keras.layers.MaxPooling2D(2)

    keras.layers.Conv2D(6,5,1,'valid',activation='relu',name='conv_1'),
    keras.layers.MaxPooling2D(2,name='pool_1'),

    keras.layers.Conv2D(12, 3, 1, 'valid', activation='relu', name='conv_2'),
    keras.layers.MaxPooling2D(2, name='pool_2'),

    # FC Layer
    # keras.layers.Flatten(),
    keras.layers.Flatten(name='flatten'),
    keras.layers.Dense(units=256, activation='relu',name='fc_1'),
    keras.layers.Dense(units=64, activation='relu',name='fc_2'),
    keras.layers.Dense(units=10, activation='softmax',name='fc_3'),

])

model.summary() # 모델의 아키텍쳐를 보여준다.


model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.01),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10,verbose = 1,
          validation_data=(x_test,y_test),
          batch_size= 100)


# p = model.predict(x_train)
# print(p)

# 1번 -> 2번 -> 3번 -> 4번
