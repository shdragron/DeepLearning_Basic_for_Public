import keras
import numpy as np

def rnn_2_simple():
    # 원-핫 인코딩된 입력 데이터
    x = [[0, 0, 0, 0, 0, 1],  # t
         [1, 0, 0, 0, 0, 0],  # e
         [0, 1, 0, 0, 0, 0],  # n
         [0, 0, 0, 0, 1, 0],  # s
         [0, 0, 1, 0, 0, 0]]  # o
    y = [0, 1, 4, 2, 3]  # 정답 레이블 인덱스
    x = np.int32([x])  # 입력 데이터를 3차원으로 변환 (배치, 타임스텝, 특성)
    y = np.int32([y])  # 타겟 레이블을 2차원으로 변환
    print(x.shape, y.shape)  # (1, 5, 6) (1, 5)

    # 알파벳 순으로 정렬된 vocab 배열
    vocab = sorted('tensor')  # ['e', 'n', 'o', 'r', 's', 't']
    vocab = np.array(vocab)
    print(vocab)

    # 모델 정의

    model = keras.Sequential([
        keras.layers.Input(shape=x.shape[1:]),              # 입력 형태 정의 (5, 6)
        keras.layers.SimpleRNN(16, return_sequences=True),  # RNN 층 (16개의 유닛)
        keras.layers.Dense(6, activation='softmax')         # 출력층 (softmax 활성화 함수)
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
    answer = 'enorst'
    for i in p_arg[0]:
        print(i, answer[i], vocab[i])

    # vocab에서 최종 예측된 문자열 출력
    print(vocab[p_arg[0]])  # 예측된 문자의 배열 출력

rnn_2_simple()
