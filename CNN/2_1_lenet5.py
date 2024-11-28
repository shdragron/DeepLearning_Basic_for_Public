# 2_1_lenet5.py

# 퀴즈: 교재에 나온 르넷5 CNN 모델을 구축해서 mnist 데이터에 대해 정확도를 구하세요.

import keras
from sklearn import preprocessing, model_selection

mnist = keras.datasets.mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist
x_train, x_test = x_train.reshape(-1,28,28,1), x_test.reshape(-1,28,28,1) # 데이터 개수, feature (이미지의 각점(픽셀) 하나가 feature이다)

x_train, x_test = x_train / 255.0, x_test / 255.0 # 정규화

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(y_train[:10])
print(type(y_train[0]))

# 우리가 만든 르넷5 모델의 파라미터 갯수를 계산하세요.

def cal_num_cost(inputs_num,channel_num, filter_size, bias):
    num_cost = inputs_num * channel_num * filter_size + bias
    return num_cost




model = keras.models.Sequential([

    keras.layers.Input(shape = x_train[0].shape),
    # convolution Layer
    # 156 = 1 * 5*5 * 6 + 6
    keras.layers.Conv2D(6,5,1,'same',activation='relu',name='conv_1'),
    keras.layers.MaxPooling2D(2,name='pool_1'),
    # 2416 = 6 * 5*5 * 16 + 16
    keras.layers.Conv2D(16, 5, 1, 'valid', activation='relu', name='conv_2'),
    keras.layers.MaxPooling2D(2, name='pool_2'),

    # FC Layer
    # keras.layers.Flatten(),
    keras.layers.Flatten(name='flatten'),
    # 48120 = 400 * 120 + 120
    keras.layers.Dense(units=120, activation='relu',name='C_5'),
    # 10164 = 120 * 84 + 84
    keras.layers.Dense(units=84, activation='relu',name='F_6'),
    #  Output     Input      weight + bias
    # (? , 10) = (? , 84) @ (84, 10) + 10
    keras.layers.Dense(units=10, activation='softmax',name='Output'),
])

model.summary() # 모델의 아키텍쳐를 보여준다.


model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100,verbose = 1,
          validation_data=(x_test,y_test),
          batch_size= 100)


# p = model.predict(x_train)
# print(p)

# 1번 -> 2번 -> 3번 -> 4번
