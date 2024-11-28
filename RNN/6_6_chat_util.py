# 6_6_chat_util.py
import numpy as np
import keras

_PAD_, _SOS_, _EOS_, _UNK_ = 0, 1, 2, 3


def load_vocab():
    f = open('chat/vocab.txt', 'r', encoding='utf-8')

    vocab = [w.strip() for w in f]
    f.close()

    return vocab


def load_vectors():
    f = open('chat/vectors.txt', 'r', encoding='utf-8')

    vectors = [[int(i) for i in w.strip().split(',')] for w in f]
    f.close()

    # print(vectors)
    return vectors


def add_pad(seq, maxlen):
    if len(seq) > maxlen:
        return seq[:maxlen]

    return seq + [_PAD_] * (maxlen - len(seq))


def make_xxy():
    vocab = load_vocab()
    vectors = load_vectors()

    questions, answers = vectors[::2], vectors[1::2]
    max_len_enc = max([len(q) for q in questions])
    max_len_dec = max([len(a) for a in answers])

    # ------------------------------------ #

    onehot = np.eye(len(vocab), dtype=np.int32)
    x_enc, x_dec, y_dec = [], [], []

    for q, a in zip(questions, answers):
        enc_in = add_pad(q, max_len_enc)
        dec_in = add_pad([_SOS_] + a, max_len_dec)
        target = add_pad(a + [_EOS_], max_len_dec)
        print(enc_in, dec_in, target)

        x_enc.append(onehot[enc_in])    # 3차원
        x_dec.append(onehot[dec_in])    # 3차원
        y_dec.append(target)            # 2차원(sparse)

    return np.float32(x_enc), np.float32(x_dec), np.float32(y_dec), vocab


def save_chat_model(x_enc, x_dec, y_dec):
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

    model.fit([x_enc, x_dec], y_dec, epochs=1000, verbose=2)
    model.save('chat/chat.keras')


if __name__ == '__main__':
    x_enc, x_dec, y_dec, vocab = make_xxy()
    save_chat_model(x_enc, x_dec, y_dec)






