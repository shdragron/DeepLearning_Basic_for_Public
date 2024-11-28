# 6_4_translation.py
import numpy as np
import keras


def make_vocab_and_index(data):
    # 퀴즈
    # 한글과 영어를 각각 모아주세요
    # kor = ''.join([k for _, k in data])
    kor = ''.join(sorted({c for _, k in data for c in k}))
    eng = ''.join(sorted({c for e, _ in data for c in e}))
    print(kor, eng)

    vocab = eng + kor + 'SEP'   # Start, End, Pad
    chr2idx = {c: i for i, c in enumerate(vocab)}
    print(chr2idx)

    return vocab, chr2idx


# ('food', '음식') -> food, S음식, 음식E
def make_xxy(data, chr2idx):
    onehot = np.eye(len(chr2idx), dtype=np.int32)
    x_enc, x_dec, y_dec = [], [], []

    for eng, kor in data:
        enc_in = [chr2idx[c] for c in eng]
        dec_in = [chr2idx[c] for c in 'S'+kor]
        target = [chr2idx[c] for c in kor+'E']
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

    model.fit([x_enc, x_dec], y_dec, epochs=10, verbose=2)
    p = model.predict([x_enc, x_dec], verbose=0)
    print(p.shape)

    p_arg = np.argmax(p, axis=2)
    print(p_arg)

    for v in p_arg:
        # print([vocab[i] for i in v[:-1]])
        print(''.join([vocab[i] for i in v[:-1]]))

    # 퀴즈
    # 'wind', 'hero'에 대해서 번역하세요
    sample = [('wind', 'PP'), ('hero', 'PP')]
    x_enc, x_dec, _ = make_xxy(sample, chr2idx)

    p = model.predict([x_enc, x_dec], verbose=0)
    p_arg = np.argmax(p, axis=2)

    for v in p_arg:
        print(''.join([vocab[i] for i in v[:-1]]))


data = [('food', '음식'), ('auto', '자동'),
        ('wind', '바람'), ('book', '도서'),
        ('hero', '영웅'), ('head', '머리')]

vocab, chr2idx = make_vocab_and_index(data)
x_enc, x_dec, y_dec = make_xxy(data, chr2idx)

show_translation(x_enc, x_dec, y_dec, vocab, chr2idx)



