# 4_3_sklearn.py
import numpy as np
from sklearn import datasets, linear_model, model_selection
import matplotlib.pyplot as plt

def sk_1():
    blobs = datasets.make_blobs(n_samples=1000, n_features=2, centers=10, random_state=10) # feature = column = variable
    print(type(blobs), len(blobs)) # <class 'tuple'> 2
    x, y = blobs
    print(x.shape, y.shape) # (100, 5) (100,)
    print(x[:5])
    print(y[:10]) # [5 9 3 2 0 9 8 5 5 3]
    print(set(y))
    # plt.scatter(x[:,0], x[:,1], c = y) # feature 2개 뽑아왔다.
    # plt.show()

    data = model_selection.train_test_split(x, y, test_size=0.2) # test, train data 분류해준다.
    x_train, x_test, y_train, y_test = data
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # 객체 생성 - 학습 - 사용
    clf = linear_model.LogisticRegression(solver='liblinear')
    # 학습
    clf.fit(x_train, y_train)
    # 검증
    print('acc: ', clf.score(x_test, y_test))

def sk_2():
    x, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=10, random_state=10)
    # data = model_selection.train_test_split(x, y, test_size=0.7) # test, train data 분류해준다.
    for i in range(0,1000,100):
        if i == 1000:
            pass
        else:
            x_train, x_test, y_train, y_test = np.concatenate((x[0:i],x[i+99:1000])), x[i:i+99], np.concatenate((y[0:i],y[i+99:1000])), y[i:i+99]
            # 객체 생성 - 학습 - 사용
            clf = linear_model.LogisticRegression(solver='liblinear')
            # 학습
            clf.fit(x_train, y_train)
            # 검증
            print('acc: ', clf.score(x_test, y_test))
def sk_3():
    x, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=10, random_state=10)
    clf = linear_model.LogisticRegression(solver='liblinear')
    print(model_selection.cross_val_score(clf, x, y, scoring='accuracy',cv=model_selection.KFold(n_splits=10)))

# sk_1()
# sk_2()
sk_3()
# cv, grid search, kaggle, preprocessing, dl




