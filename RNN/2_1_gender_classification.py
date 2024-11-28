# 2_1_gender_classification.py
import random
import nltk
import string

# nltk.download('names')


# 퀴즈
# names 코퍼스로 아래와 같은 형식의 데이터를 만드세요
# [('kim', 'male'), ('im', 'female'), ('nam', 'male'), ...]
def make_labeled_names():
    print(nltk.corpus.names.fileids())  # ['female.txt', 'male.txt']

    males = nltk.corpus.names.words('male.txt')
    females = nltk.corpus.names.words('female.txt')
    print(males)

    male_data = [(name.lower(), 'male') for name in males]
    female_data = [(name.lower(), 'female') for name in females]
    print(male_data[:3])

    random.seed(41)
    random.shuffle(male_data)
    random.shuffle(female_data)

    data = male_data + female_data
    print(len(data), len(male_data), len(female_data))

    return data


def make_feature_1(name):
    feature = {'last_letter': name[-1]}
    return feature


def make_feature_2(name):
    feature = {'last_letter': name[-1],
               'first_letter': name[0]}
    return feature


def make_feature_3(name):
    feature = {'last_letter': name[-1],
               'first_letter': name[0]}

    for ch in string.ascii_lowercase:
        feature['has_{}'.format(ch)] = (ch in name)
        feature['count_{}'.format(ch)] = name.count(ch)

    return feature


def make_feature_4(name):
    feature = {'suffix1': name[-1],
               'suffix2': name[-2]}
    return feature


# 퀴즈
# 여러분이 생각하는 피처를 추가해서 75.21% 이상의 결과를 만들어 보세요
def make_feature_5(name):
    # 'vowels': sum(['aeiou'.count(ch) for ch in name])
    # 남녀 이름에 자주 등장하는 패턴 피처 추가
    feature = {'suffix1': name[-1],
               'suffix2': name[-2],
               'first_letter': name[0],
               'length': len(name),
               'second_letter': name[1]}
    return feature


def make_feature_6(name):
    feature = {'suffix1': name[-1],
               'suffix2': name[-2:],
               'suffix3': name[-3:],}
    return feature


# labeled_names: (name, gender)
def gender_classification(labeled_names, feature_func):
    data = [(feature_func(name), gender) for name, gender in labeled_names]
    print(data[:3])

    # random.shuffle(data)
    # train_set, test_set = data[:-1000], data[-1000:]

    test_set = data[:500] + data[-500:]         # 1000
    train_set = data[500:-500]                  # 6944

    clf = nltk.NaiveBayesClassifier.train(train_set)
    print('acc :', nltk.classify.accuracy(clf, test_set))

    print(clf.classify(feature_func('trinity')))
    print(clf.classify(feature_func('neo')))

    # 퀴즈
    # 잘못 예측한 이름에 대해서 알고 싶어요
    test_data = labeled_names[500:-500]
    males_wrong, females_wrong = [], []
    for name, gender in test_data:
        # print(name, gender)

        result = clf.classify(feature_func(name))
        # print(name, result)

        if gender != result:
            if result == 'male':
                males_wrong.append(name)
            else:
                females_wrong.append(name)

    print('남자를 여자로 판단 :', females_wrong[:5])
    print('여자를 남자로 판단 :', males_wrong[:5])
    clf.show_most_informative_features(25)



names = make_labeled_names()

# gender_classification(names, make_feature_1)
# gender_classification(names, make_feature_2)
# gender_classification(names, make_feature_3)      # 0.758
# gender_classification(names, make_feature_4)
gender_classification(names, make_feature_5)        # 0.773
# gender_classification(names, make_feature_6)




