# 4_2_googlenet.py

# 퀴즈: googlenet 아키텍쳐로부터 stem일부만 구현해보세요.

import keras

input = keras.layers.Input(shape=[299,299,3])


Conv_1 = keras.layers.Conv2D(32, 3,strides=2,padding='valid',activation='relu', name = '1')(input)
Conv_2 = keras.layers.Conv2D(32, 3,strides=1,activation='relu',padding='valid', name = '2')(Conv_1)
Conv_3 = keras.layers.Conv2D(64, 3,padding='same',strides=1,activation='relu', name = '3')(Conv_2)

Pool = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='valid' ,name = '1_1')(Conv_3)
Conv_4 = keras.layers.Conv2D(96, 3, strides=2, padding='valid' ,activation='relu', name = '1_2')(Conv_3)

filter_c = keras.layers.concatenate([Conv_4,Pool], name = 'cncat')

Conv_5 = keras.layers.Conv2D(64, 1, strides=1 ,activation='relu',padding='same', name = '2_1_1')(filter_c)
Conv_6 = keras.layers.Conv2D(96, 3,padding='valid', strides=1 ,activation='relu', name = '2_1_2')(Conv_5)

Conv_7 = keras.layers.Conv2D(64, 1, strides=1 ,activation='relu',padding='same', name = '2_2_1')(filter_c)
Conv_8 = keras.layers.Conv2D(64, [7,1], strides=1 ,activation='relu',padding='same', name = '2_2_2')(Conv_7)
Conv_9 = keras.layers.Conv2D(64, [1,7], strides=1 ,activation='relu',padding='same', name = '2_2_3')(Conv_8)
Conv_10 = keras.layers.Conv2D(96, 3, padding='valid',strides=1 ,activation='relu', name = '2_2_4')(Conv_9)

filter_c_1 = keras.layers.concatenate([Conv_10,Conv_6], name = 'cncat_1')

Conv_11 = keras.layers.Conv2D(192, 3, strides=2, padding='valid' ,activation='relu', name = '3_1')(filter_c_1)
Pool_1 = keras.layers.MaxPooling2D(2, strides=2, padding='valid', name = '3_2')(filter_c_1)

filter_c_2 = keras.layers.concatenate([Pool_1,Conv_11], name = 'cncat_2')

model = keras.models.Model(inputs=input, outputs=filter_c_2)
model.summary()


