# 5_3_naver_movie.py
import csv
import re
import time
import numpy as np


# 퀴즈
# ratings_test.txt 파일을 읽고 x, y를 반환하는 함수를 만드세요
def make_xy(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    f.readline()

    x, y = [], []
    for _, review, label in csv.reader(f, delimiter='\t'):
        x.append(clean_str(review).split())
        y.append(int(label))

    f.close()
    return x, np.reshape(-1, 1)


# 김윤 박사 깃헙
def clean_str(string):
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


start = time.time()
x_train, y_train = make_xy('data/ratings_train.txt')
x_test, y_test = make_xy('data/ratings_test.txt')
print('소요 시간 :', time.time() - start)





