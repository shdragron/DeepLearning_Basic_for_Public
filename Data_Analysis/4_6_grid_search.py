# 4_6_grid_search.py
# 해당 모델에서 어떤 매개변수 값이 잘나오는가?
import pandas as pd
from sklearn import datasets, model_selection, svm
import numpy as np

# 학습 및 테스트 함수
def simple(x_train, y_train, x_test, y_test):
    best_score = 0.0
    for gamma in ( 1, 10, 100):
        for C in (0.001, 0.01, 0.1, 1, 10, 100):
            svc = svm.SVC(C=C, gamma=gamma)
            svc.fit(x, y)
            score = svc.score(x_test, y_test)
            print('acc: ', score)  # acc:  0.9733333333333334
            # svc = svm.SVC()
        # 가장 확률이 좋은 파라미터 구하기
            if (score > best_score):
                best_score = score
                best_params = {'gamma': gamma, 'C': C}
    print('best: ', best_score)
    print('best_params: ', best_params)
    # 베스트 파라미터로 학습후 출력
    svc = svm.SVC(**best_params)
    svc.fit(x_train, y_train)
    score = svc.score(x_test, y_test)
    print('acc: ', score)  # acc:  0.9733333333333334
    # svc = svm.SVC()


# simple의 문제점 한번 학습해서 나온 결과값으로 신뢰도가 떨어진다.

def cross_validation(x_train, y_train, x_test, y_test):
    best_score = 0.0
    for gamma in ( 1, 10, 100):
        for C in (0.001, 0.01, 0.1, 1, 10, 100):
            svc = svm.SVC(C=C, gamma=gamma)
            scores = model_selection.cross_val_score(svc,x_train, y_train, cv = 5)
            score = np.mean(scores) # 5개 배열로 결과가 나와서 수정 평균을 본다.
            print('acc: ', score)
            # svc = svm.SVC()
        # 가장 확률이 좋은 파라미터 구하기
            if (score > best_score):
                best_score = score
                best_params = {'gamma': gamma, 'C': C}
    print('best: ', best_score)
    print('best_params: ', best_params)
    # 베스트 파라미터로 학습후 출력
    svc = svm.SVC(**best_params)
    svc.fit(x_train, y_train)
    score = svc.score(x_test, y_test)
    print('acc: ', score)
    # svc = svm.SVC()

# 단점: 반복문이 너무 많다.

def grid_search(x_train, y_train, x_test, y_test):
    grid_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
                   }
    gs = model_selection.GridSearchCV(svm.SVC(**grid_params), param_grid= grid_params, cv=5)
    gs.fit(x_train, y_train)

    print('test score:', gs.score(x_test, y_test))
    print('best params: ', gs.best_params_)
    print('best score: ', gs.best_score_)

    print(gs.cv_results_) # 이 코드는 데이터 형식 자체가 가시성 낮다.

    # 판다스로 데이터 가시성 높인다.
    df = pd.DataFrame(gs.cv_results_)
    print(df)
    print()

    print(df.T)

# 현실적으로 사용하기 어렵다. 경우의 수가 너무 많아.

def random_search(x_train, y_train, x_test, y_test):
    grid_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
                   }
    gs = model_selection.RandomizedSearchCV(svm.SVC(**grid_params), param_distributions=grid_params, cv=5)
    gs.fit(x_train, y_train)

    print('test score:', gs.score(x_test, y_test))
    print('best params: ', gs.best_params_)
    print('best score: ', gs.best_score_)

    print(gs.cv_results_)  # 이 코드는 데이터 형식 자체가 가시성 낮다.

    # 판다스로 데이터 가시성 높인다.
    df = pd.DataFrame(gs.cv_results_)
    print(df)
    print()

    print(df.T)

# 데이터가 매우 클경우 랜덤서치를 이용하여 최적의 하이퍼파라미터를 얻는다.

x, y = datasets.load_iris(return_X_y=True)
print(x.shape, y.shape)  # (150, 4) (150,)

# 데이터 분류(test, train)
data = model_selection.train_test_split(x, y, test_size=0.3)
x_train, x_test, y_train, y_test = data

# 학습 및 테스트
# simple(x_train, y_train, x_test, y_test)
# cross_validation(x_train, y_train, x_test, y_test)
# grid_search(x_train, y_train, x_test, y_test)
random_search(x_train, y_train, x_test, y_test)












