# 6_2_word2vec_king.py
import numpy as np
import keras
import matplotlib.pyplot as plt


def make_vocab_and_vectors():
    corpus = ['king is a strong man',
              'queen is a wise woman',
              'boy is a young man',
              'girl is a young woman',
              'prince is a young king',
              'princess is a young queen',
              'man is strong',
              'woman is pretty',
              'prince is a boy will be king',
              'princess is a girl will be queen']

    # 퀴즈
    # 코퍼스로부터 불용어를 제거한 2차원 토큰 리스트를 만드세요
    # [['king', 'strong', 'man'], ['queen', 'wise', 'woman'], ...]
    stopwords = ['is', 'a', 'will', 'be']

    tokens = [s.split() for s in corpus]
    # [['king', 'is', 'a', 'strong', 'man'], ...]

    tokens = [[w for w in t if w not in stopwords] for t in tokens]
    # [['king', 'strong', 'man'], ...]

    # 퀴즈
    # 토큰 리스트로부터 단어장을 만드세요 (vocab)
    # vocab = []
    # for t in tokens:
    #     for w in t:
    #         print(w)
    #         if w not in vocab:
    #             vocab.append(w)

    # vocab = [w for t in tokens for w in t]
    # vocab = set(vocab)

    vocab = {w for t in tokens for w in t}
    vocab = sorted(vocab)
    print(vocab)
    # ['boy', 'girl', 'king', 'man', 'pretty', 'prince', 'princess', 'queen', 'strong', 'wise', 'woman', 'young']

    vectors = [[vocab.index(w) for w in t] for t in tokens]
    print(vectors)
    # [[2, 8, 3], [7, 9, 10], ...]

    return vocab, vectors


def extract_surrounds(tokens, center, window_size):
    start = max(0, center-window_size)
    end = min(len(tokens), center+window_size+1)
    return [tokens[i] for i in range(start, end) if i != center]


def make_xy(vectors, vocab, window_size, is_skipgram):
    xx, yy = [], []
    for tokens in vectors:
        for center in range(len(tokens)):
            surrounds = extract_surrounds(tokens, center, window_size)
            # print(surrounds)

            if is_skipgram:
                for p in surrounds:
                    # print(center, tokens[center], p)
                    xx.append(tokens[center])
                    yy.append(p)
            else:
                xx.append(surrounds)
                yy.append(tokens[center])

    # print(xx)
    # cbow [[8, 3], [2, 3], ...]
    # [[0. 0. 0. .5 0. 0. 0. 0. .5 0. 0. 0.]
    #  [0. 0. .5 .5 0. 0. 0. 0. 0. 0. 0. 0.]
    # skipgram [2, 2, 8, ...]
    # [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]

    x = np.zeros([len(xx), len(vocab)], dtype=np.float32)
    # print(x)

    for i, p in enumerate(xx):
        if is_skipgram:
            # print(i, p)
            x[i, p] = 1
        else:
            # v = 1 / len(p)
            # for j in p:
            #     x[i, j] = v
            x[i, p] = 1 / len(p)

            # onehot = [[int(j == k) for k in range(len(vocab))] for j in p]
            # x[i] = np.mean(onehot, axis=0)
            # print(onehot)

    # print(x)
    return x, np.int32(yy)


class EpochStep(keras.callbacks.Callback):
    def __init__(self, step=500):
        super().__init__()
        self.step = step

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if epoch % self.step == 0:
            print(epoch, logs)


vocab, vectors = make_vocab_and_vectors()

x, y = make_xy(vectors, vocab, window_size=2, is_skipgram=False)

model = keras.Sequential([
    keras.layers.Input(shape=[len(vocab)]),                 # (?, 12)
    keras.layers.Dense(2),                                  # (?, 2) = (?, 12) @ (12, 2)
    keras.layers.Dense(len(vocab), activation='softmax'),   # (?, 12) = (?, 2) @ (2, 12)
])
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

epoch_step = EpochStep(step=500)
model.fit(x, y, epochs=20000, verbose=0, callbacks=[epoch_step])

layer = model.get_layer(index=0)
print(layer)

w, b = layer.get_weights()
print(w, b)

# print(vocab[3], w[3])

for token, (x1, x2) in zip(vocab, w):
    plt.plot(x1, x2, 'ro')
    plt.text(x1, x2, token)

plt.show()

