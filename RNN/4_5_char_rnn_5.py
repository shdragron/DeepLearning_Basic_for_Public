# 4_5_char_rnn_5.py

import keras
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# 길이가 다른 단어가 들어온다.
def rnn_5(words):
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
    # print(''.join(vocab[p_arg[0]]))

    for i in range(len(words)):
        pp = p_arg[i]
        valid = len(words[i]) - 1
        qq = pp[:valid]
        qq = ''.join(vocab[qq])
        print(qq)

def make_xy(words):


    bin = LabelBinarizer()
    bin.fit(list(''.join(words))) # 'tensorcoffeedonuts'

    max_len = max([ len(w) for w in words])

    x , y = [], []

    for i, w in enumerate(words):
        w += ' '* (max_len - len(w))
        onehot = bin.transform(list(w))
        x_1 = onehot[:-1]
        y_1 = np.argmax(onehot[1:],axis=1)
        x.append(x_1)
        y.append(y_1)


    x = np.array(x)
    y = np.array(y)

    return x, y, bin.classes_


# make_xy('tensor')
# rnn_3('rainbow')
rnn_5(['chat','americano','donuts'])