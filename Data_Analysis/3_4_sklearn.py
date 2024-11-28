# 3_4_sklearn.py

from sklearn import datasets, svm
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

def sk_1():
    # load, make, fetch
    iris = datasets.load_iris()
    print(iris)
    print(iris.keys)
    print(iris['data']) # [5.9 3.  5.1 1.8] Sepal: length width, Petal: length width

    print(iris['target_names']) # label
    print(iris['feature_names'])
    print(iris['target']) # target_names의 index화 0은 'setosa'

def sk_2():
    iris = datasets.load_iris()

    df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    print(df)

    # scatter_matrix(df, c=iris['target']) #인덱스를 이용해서 색상을 구분해준다.
    # scatter_matrix(df, c=iris['target'], hist_kwds={'bins': 150}) # hist_kwds 얇기
    scatter_matrix(df, c=iris['target'], hist_kwds={'bins': 150}, cmap = 'jet') # hist_kwds 얇기
    # cmap은 색상 설정
    plt.show()
def sk_3():
    digits = datasets.load_digits()
    print(digits.keys())
    # ['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR']
    print(digits['target_names']) # 0~9까지 label
    print(digits['feature_names']) # 28x28 이지만 8x8 로 축소된 버전
    print(len(digits['target'])) # 7만장 중 1797개
    print(digits['data'].shape) # (1797, 64) 이미지 당 픽셀 당  모노톤 값의 평균
    print(digits['images'].shape) # 이미지 당 픽셀 당  모노톤 값 (1797, 8, 8)

    # 1. 객체 생성
    clf = svm.SVC() # 모델 -> 하이퍼파라미터 튜닝
    # 2. 학습
    clf.fit(digits['data'], digits['target']) # 학습
    # 3. 사용
    print(clf.score(digits['data'], digits['target'])) # 0.996661101836394
    p = clf.predict(digits['data'])
    print(len(p)) # 1797

    print(digits['target'][:10]) # 정답 [0 1 2 3 4 9 6 7 8 9]
    print(p[:10]) # 예측값 [0 1 2 3 4 9 6 7 8 9]

    per = np.mean(np.equal(p, digits['target'])) # boolean 값은 0과 1이다.
    print("acc: ",per) # acc: 0.996661101836394

    # 퀴즈: svm을 사용하여, 70%로 데이터로 학습 시키고 30%에 대해 정확도를 구하세요.

def sk_4():
    digits = datasets.load_digits()
    # 1. 객체 생성
    clf = svm.SVC()
    # 2. 학습
    p30 = int((1797/100)*30)
    clf.fit(digits['data'][:p30], digits['target'][:p30])
    # 3. 사용
    print(clf.score(digits['data'][p30:], digits['target'][p30:])) # 0.9451510333863276
    p = clf.predict(digits['data'])
    print(p) # [0 1 2 ... 8 9 8]


    # 인공지능 종류
    # 1. 지도학습
    # 수학 선생님이 수학을 교육
    # 2. 비지도학습
    # 수학선생님이 국어를 교육
    # 3. 비지도학습 반 지도학습 반 -> 강화학습
    # 강화학습

# sk_1()
# sk_2()
# sk_3()
sk_4()

