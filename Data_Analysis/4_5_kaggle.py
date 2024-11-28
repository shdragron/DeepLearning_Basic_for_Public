# 4_4_kaggle.py
# 어떤 모델이 해당 데이터에서 적합할까?
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# 퀴즈
# 캐글 leaf 컨페티션에서 most votes를 받은 소스코드를 옮겨와서 구동하세요
train = pd.read_csv('./data/leaf_classification/train.csv')
test = pd.read_csv('./data/leaf_classification/test.csv')


def encode(train, test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)  # encode species strings
    classes = list(le.classes_)  # save column names for submission
    test_ids = test.id  # save test ids for submission

    train = train.drop(['species', 'id'], axis=1)
    test = test.drop(['id'], axis=1)

    return train, labels, test, test_ids, classes


train, labels, test, test_ids, classes = encode(train, test)
print(train.head(1))

sss = StratifiedShuffleSplit(10, test_size=0.2, random_state=23)

for train_index, test_index in sss.split(train, labels):
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="rbf", C=0.025, probability=True),
        NuSVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        # GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

    # Logging for Visual Comparison
    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)

    for clf in classifiers:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        print('****Results****')
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))

        train_predictions = clf.predict_proba(X_test)
        ll = log_loss(y_test, train_predictions)
        print("Log Loss: {}".format(ll))

        log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
        # log = log.append(log_entry)

    print("=" * 30)

# Predict Test Set
favorite_clf = LinearDiscriminantAnalysis()
favorite_clf.fit(X_train, y_train)
test_predictions = favorite_clf.predict_proba(test)

# Format DataFrame
submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export Submission
submission.to_csv('./data/leaf_classification/submission.csv', index = False)
print(submission.tail())