# 4_3_word_rnn.py

import keras
import numpy as np
from sklearn.preprocessing import LabelBinarizer

motto = 'life is full of ups and downs'

tokens = motto.split()
print(tokens)

# 4_3_char_rnn_3.py



def rnn_3(words):
    x, y, vocab = make_xy(words)

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
    print(vocab[p_arg[0]])  # 예측된 문자의 배열 출력
    print(vocab[p_arg])
    print(''.join(vocab[p_arg[0]]))


def make_xy(words):

    bin = LabelBinarizer()
    one_hot = bin.fit_transform(words)
    print(one_hot)

    x = one_hot[:-1]
    y = np.argmax(one_hot[1:],axis=1)
    x = x[np.newaxis]
    y = y[np.newaxis]

    print(y)

    # 알파벳 순으로 정렬된 vocab 배열
    word = list(set(words))
    vocab = sorted(words)
    vocab = np.array(vocab)
    print(vocab)

    return x, y, vocab

# make_xy('tensor')
# rnn_3('rainbow')
rnn_3(tokens)