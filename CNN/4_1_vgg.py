# 4_1_vgg.py
import keras
# 퀴즈: VGG16 모델을 만들고 파라미터 갯수가 1.38억개가 맞는지 확인해보세요.



model = keras.models.Sequential([

    keras.layers.Input(shape = [224,224,3]),


    keras.layers.Conv2D(64,3,1,'same',activation='relu',name='conv_1'),
    keras.layers.Conv2D(64, 3, 1, 'same', activation='relu', name='conv_1_1'),
    keras.layers.MaxPooling2D(2,name='pool_1'),


    keras.layers.Conv2D(128, 3, 1, 'same', activation='relu', name='conv_2'),
    keras.layers.Conv2D(128, 3, 1, 'same', activation='relu', name='conv_2_2'),
    keras.layers.MaxPooling2D(2, name='pool_2'),

    keras.layers.Conv2D(256, 3, 1, 'same', activation='relu', name='conv_3_1'),
    keras.layers.Conv2D(256, 3, 1, 'same', activation='relu', name='conv_3_2'),
    keras.layers.Conv2D(256, 3, 1, 'same', activation='relu', name='conv_3_3'),
    keras.layers.MaxPooling2D(2, name='pool_3'),

    keras.layers.Conv2D(512, 3, 1, 'same', activation='relu', name='conv_4_1'),
    keras.layers.Conv2D(512, 3, 1, 'same', activation='relu', name='conv_4_2'),
    keras.layers.Conv2D(512, 3, 1, 'same', activation='relu', name='conv_4_3'),
    keras.layers.MaxPooling2D(2, name='pool_4'),

    keras.layers.Conv2D(512, 3, 1, 'same', activation='relu', name='conv_4_4'),
    keras.layers.Conv2D(512, 3, 1, 'same', activation='relu', name='conv_4_5'),
    keras.layers.Conv2D(512, 3, 1, 'same', activation='relu', name='conv_4_6'),
    keras.layers.MaxPooling2D(2, name='pool_5'),

    # FC Layer

    # keras.layers.Flatten(name='flatten'),
    #
    # keras.layers.Dense(units=4096, activation='relu',name='F_1'),
    #
    # keras.layers.Dense(units=4096, activation='relu',name='F_2'),
    #
    # keras.layers.Dense(units=1000, activation='softmax',name='Output'),

    # FC -> Conv 같은 것

    keras.layers.Conv2D(4096,[7,7],1,activation='relu',name='fc_1'),
    keras.layers.Conv2D(4096,[1,1],1,activation='relu',name='fc_2'),
    keras.layers.Conv2D(1000,[1,1],1,activation='softmax',name='output'),
    keras.layers.Flatten(name='Flatten'),

])

model.summary() # 모델의 아키텍쳐를 보여준다.



# p = model.predict(x_train)
# print(p)

# 1번 -> 2번 -> 3번 -> 4번
