# 6_5_translation_word.py
import numpy as np
import keras


def make_vocab_and_index(data):
    kor = sorted({c for _, k in data for c in k.split()})
    eng = sorted({c for e, _ in data for c in e.split()})
    print(kor)
    print(eng)

    vocab = eng + kor + ['_SOS_', '_EOS_', '_PAD_']
    chr2idx = {c: i for i, c in enumerate(vocab)}
    print(chr2idx)

    return vocab, chr2idx


# ('food', '음식') -> food, S음식, 음식E
def make_xxy(data, chr2idx):
    onehot = np.eye(len(chr2idx), dtype=np.int32)
    x_enc, x_dec, y_dec = [], [], []

    for eng, kor in data:
        enc_in = [chr2idx[c] for c in eng.split()]
        dec_in = [chr2idx[c] for c in ['_SOS_'] + kor.split()]
        target = [chr2idx[c] for c in kor.split() + ['_EOS_']]
        print(enc_in, dec_in, target)

        x_enc.append(onehot[enc_in])    # 3차원
        x_dec.append(onehot[dec_in])    # 3차원
        y_dec.append(target)            # 2차원(sparse)

    return np.float32(x_enc), np.float32(x_dec), np.float32(y_dec)


def show_translation(x_enc, x_dec, y_dec, vocab, chr2idx):
    # 인코더
    enc_in = keras.layers.Input(x_enc.shape[1:])
    _, enc_state = keras.layers.SimpleRNN(128, return_state=True)(enc_in)

    # 디코더
    dec_in = keras.layers.Input(x_dec.shape[1:])
    output = keras.layers.SimpleRNN(128, return_sequences=True)(dec_in, initial_state=enc_state)
    output = keras.layers.Dense(len(vocab), activation='softmax')(output)

    model = keras.Model([enc_in, dec_in], output)
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit([x_enc, x_dec], y_dec, epochs=100, verbose=2)

    sample = [('how many trees are there', '_PAD_ _PAD_ _PAD_ _PAD_'),
              ('where do wind come from', '_PAD_ _PAD_ _PAD_ _PAD_')]
    x_enc, x_dec, _ = make_xxy(sample, chr2idx)

    p = model.predict([x_enc, x_dec], verbose=0)
    p_arg = np.argmax(p, axis=2)

    for v in p_arg:
        print(' '.join([vocab[i] for i in v[:-1]]))


# 퀴즈
# 앞에서 배운 char 번역을 word 번역 코드로 수정하세요
data = [('did you eat some food', '너는 음식을 좀 먹었니'),
        ('how many trees are there', '그곳에 나무가 얼마나 있니'),
        ('i really like blue color', '나는 파랑색을 진짜 좋아해'),
        ('today is happy christmas eve', '오늘은 행복한 성탄절 전야라네'),
        ('where do wind come from', '바람은 어디에서 오는 것일까'),
        ('we really need the hero', '우리에겐 진짜 영웅이 필요해')]

vocab, chr2idx = make_vocab_and_index(data)
x_enc, x_dec, y_dec = make_xxy(data, chr2idx)

show_translation(x_enc, x_dec, y_dec, vocab, chr2idx)
