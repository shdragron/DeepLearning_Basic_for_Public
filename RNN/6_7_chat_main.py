# 6_7_chat_main.py
import numpy as np
import keras
import nltk
chat_util = __import__('6_6_chat_util')


def decode_prediction(seq, vocab, is_question):
    seq = list(seq)         # ndarray -> list

    pos = seq.index(chat_util._EOS_) if chat_util._EOS_ in seq else len(seq)
    seq = seq[:pos]

    result = ' '.join(vocab[i] for i in seq)
    result = result.replace(vocab[chat_util._PAD_], '')
    print('여우 :' if is_question else '왕자 :', result)


def load_and_predict():
    x_enc, x_dec, _, vocab = chat_util.make_xxy()
    model = keras.models.load_model('chat/chat.keras')

    p = model.predict([x_enc, x_dec], verbose=0)
    p_arg = np.argmax(p, axis=2)

    for q, a, pred in zip(x_enc, x_dec, p_arg):
        # print(q.shape, a.shape, pred)

        question = np.argmax(q, axis=1)
        answer = np.argmax(a, axis=1)
        # print(question, answer, pred)

        decode_prediction(question, vocab, is_question=True)
        decode_prediction(answer[1:], vocab, is_question=False)
        decode_prediction(pred, vocab, is_question=False)
        print()


def talk_with_bot():
    vocab = chat_util.load_vocab()
    vectors = chat_util.load_vectors()

    questions, answers = vectors[::2], vectors[1::2]
    max_len_enc = max([len(q) for q in questions])
    max_len_dec = max([len(a) for a in answers])

    # ------------------------------------ #
    onehot = np.eye(len(vocab), dtype=np.float32)

    x_dec = chat_util.add_pad([chat_util._SOS_], max_len_dec)   # 1차원
    x_dec = onehot[x_dec]                                       # 2차원
    x_dec = x_dec[np.newaxis]                                   # 3차원
    # print(x_dec.shape)            # (1, 9, 164)

    model = keras.models.load_model('chat/chat.keras')

    while True:
        line = input('여우 : ')

        if '끝' in line:
            break

        x1 = nltk.tokenize.regexp_tokenize(line, r'\w+')
        x2 = [vocab.index(t) if t in vocab else chat_util._UNK_ for t in x1]
        x3 = chat_util.add_pad(x2, max_len_enc)

        x_enc = onehot[x3]
        x_enc = x_enc[np.newaxis]

        p = model.predict([x_enc, x_dec], verbose=0)
        p_arg = np.argmax(p, axis=2)                # (1, 9, 164) -> (1, 9)
d
        decode_prediction(p_arg[0], vocab, is_question=False)
        print()


# load_and_predict()
talk_with_bot()
