# 4_6_char_rnn_final.py


import keras
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# 엄청 긴 문장을 동일 한 길이의 시퀀스로 나눠서 원핫 백터 및 단어장으로 변환
def rnn_final(sentence, seq_length):
    x, y, vocab = make_xy(sentence, seq_length)

    # 모델 정의

    model = keras.Sequential([
        keras.layers.Input(shape=x.shape[1:]),              # 입력 형태 정의 (5, 6)
        keras.layers.SimpleRNN(16, return_sequences=True),  # RNN 층 (16개의 유닛)
        keras.layers.Dense(len(vocab), activation='softmax')         # 출력층 (softmax 활성화 함수)
    ])
    model.summary()


    # 모델 컴파일 (Adam 옵티마이저, sparse_categorical_crossentropy 손실 함수 사용)
    model.compile(optimizer=keras.optimizers.Adam(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    # 모델 학습 (10 에폭 동안 학습)
    model.fit(x, y, epochs=10, verbose=2)

    # 예측
    p = model.predict(x, verbose=0)
    print(p)
    print(p.shape)  # (1, 5, 6)

    # argmax를 사용해 예측된 클래스 인덱스 추출
    p_arg = np.argmax(p, axis=2)
    print(p_arg)  # 예측된 인덱스

    # 예측 정확도 계산
    print('acc :', np.mean(p_arg == y))

    # 예측된 결과를 vocab에서 문자로 변환

    # vocab에서 최종 예측된 문자열 출력
    print('I',''.join(vocab[p_arg[0]]), end = '', sep = '')
    for i in range(1,len(p_arg)):
        pp = p_arg[i]
        print((vocab[pp][-1]), end='')

    print('\n')
    print(sentence)

def make_xy(sentence, seq_length):


    bin = LabelBinarizer()
    onehot = bin.fit_transform(list((sentence))) # 'tensorcoffeedonuts'

    print(len(sentence))
    print(onehot.shape)
    print(onehot)

    x, y = [], []
    for i in range(len(sentence)-seq_length):
        xy = (onehot[i : i + seq_length + 1])
        xx = xy[:-1]
        yy = np.argmax(xy[1:], axis=1)

        x.append(xx)
        y.append(yy)

    x = np.array(x)
    y = np.array(y)

    return x, y, bin.classes_


# make_xy('tensor')
# rnn_3('rainbow')
motto = "If you want to build a ship, don't drum up people to collect wood"\
        "and don't assign them tasks and work, but rather teach them"\
        "to long for the endless immensity of the sea."
rnn_final(motto, seq_length=20)
# make_xy(motto, 20)
# If you want to build -> seq_len=5
# If yo
#      u wan
# -> 겹치는 영역이 없다.
# If yo
#     ou wan
# -> 할 것

