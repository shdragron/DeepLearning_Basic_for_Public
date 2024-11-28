# 3_4_preprocessing.py
import numpy as np
from sklearn import preprocessing


def show_label_encoder():
    cities = ['figi', 'bali', 'rome', 'bali', 'bali']

    enc = preprocessing.LabelEncoder()

    enc.fit(cities)
    t = enc.transform(cities)
    # t = enc.fit_transform(cities)
    print(t)
    print(enc.classes_)

    # 퀴즈
    # t를 원래 데이터로 복원하세요 (디코딩)
    for v in t:
        print(v, enc.classes_[v])

    print([enc.classes_[v] for v in t])
    print(enc.classes_[t])
    print(enc.classes_[[0, 2, 0, 0]])

    print(enc.transform(['figi', 'rome']))

6_1
def show_label_binarizer():
    cities = ['figi', 'bali', 'rome', 'bali', 'bali']

    bin = preprocessing.LabelBinarizer()

    bin.fit(cities)
    t = bin.transform(cities)
    print(t)
    print(bin.classes_)

    # 퀴즈
    # t를 원래 데이터로 복원하세요 (디코딩)
    r = np.argmax(t, axis=1)
    print(bin.classes_[r])


# show_label_encoder()
show_label_binarizer()









