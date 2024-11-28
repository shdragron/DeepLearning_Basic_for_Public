# 4_1_char_rnn_1.py
import keras
import numpy as np

# character, word, sentence
# tensor
# x: tenso
# y: ensor


# 퀴즈
# 아래 데이터에 대해 동작하는 딥러닝 모델을 만드세요
def rnn_1_dense():
    x = [[1, 0, 0, 0, 0, 0],  # t
         [0, 1, 0, 0, 0, 0],  # e
         [0, 0, 1, 0, 0, 0],  # n
         [0, 0, 0, 1, 0, 0],  # s
         [0, 0, 0, 0, 1, 0]]  # o
    y = [[0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]]
    x = np.int32(x)
    y = np.int32(y)

    model = keras.Sequential([
        keras.layers.Input(shape=[6]),  # 입력 크기가 6인 입력층 정의
        keras.layers.Dense(6, activation='softmax')  # 출력층: 6개의 노드, softmax 활성화 함수 사용
    ])
    model.summary()  # 모델 구조 요약 출력

    model.compile(optimizer=keras.optimizers.Adam(0.1),  # Adam 옵티마이저 사용, 학습률 0.1
                  loss=keras.losses.categorical_crossentropy,  # 다중 클래스 분류를 위한 손실 함수
                  metrics=['acc'])  # 모델 성능을 accuracy로 평가

    model.fit(x, y, epochs=10, verbose=2)  # 모델 학습: 10번 반복, verbose=2로 학습 과정 출력


# 퀴즈
# dense 버전의 코드를 sparse 버전으로 수정하세요
def rnn_1_sparse():
    x = [[1, 0, 0, 0, 0, 0],  # t
         [0, 1, 0, 0, 0, 0],  # e
         [0, 0, 1, 0, 0, 0],  # n
         [0, 0, 0, 1, 0, 0],  # s
         [0, 0, 0, 0, 1, 0]]  # o
    y = [1, 2, 3, 4, 5]  # 각 원-핫 인코딩 벡터에 해당하는 인덱스
    x = np.int32(x)
    y = np.int32(y)

    model = keras.Sequential([
         keras.layers.Input(shape=[6]),  # 입력 크기가 6인 입력층 정의
         keras.layers.Dense(6, activation='softmax')  # 출력층: 6개의 노드, softmax 활성화 함수 사용
    ])
    model.summary()  # 모델 구조 요약 출력

    model.compile(optimizer=keras.optimizers.Adam(0.1),  # Adam 옵티마이저 사용, 학습률 0.1
                  loss=keras.losses.sparse_categorical_crossentropy,  # 손실 함수로 sparse categorical crossentropy 사용
                  metrics=['acc'])  # 모델 성능을 accuracy로 평가

    model.fit(x, y, epochs=10, verbose=2)  # 모델 학습: 10번 반복, verbose=2로 학습 과정 출력



# rnn_1_dense()
rnn_1_sparse()
