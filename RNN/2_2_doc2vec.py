# 2_2_doc2vec.py
import random
import nltk
import collections
import numpy as np


# nltk.download('movie_reviews')


# 퀴즈
# 영화 리뷰 코퍼스로부터 빈도가 가장 높은 2000개의 토큰을 반환하는 함수를 만드세요
def make_vocab(size=2000):
    ids = nltk.corpus.movie_reviews.fileids()
    # print(ids)
    # print(sum(['neg' in name for name in ids]))
    # print(sum(['pos' in name for name in ids]))

    # print(nltk.corpus.movie_reviews.raw('neg/cv000_29416.txt'))

    # tokens = nltk.corpus.movie_reviews.words()
    # print(len(tokens))

    reviews = nltk.corpus.movie_reviews.raw()
    reviews = reviews.lower()
    tokens = nltk.tokenize.regexp_tokenize(reviews, r'\w+')
    # print(len(tokens))      # 1336782

    stopwords = nltk.corpus.stopwords.words('english')

    tokens = [t for t in tokens if t not in stopwords]
    # print(len(tokens))      # 708572

    freq = collections.Counter(tokens)
    # print(freq.most_common(5))
    # print(type(freq.most_common(5)))

    return [w for w, _ in freq.most_common(size)]


def make_feature(doc_tokens, vocab):
    feature, uniques = {}, set(doc_tokens)
    for v in vocab:
        feature['has_'+v] = (v in uniques)

    return feature


vocab = make_vocab()

# doc_tokens = nltk.corpus.movie_reviews.words('neg/cv000_29416.txt')
# print(doc_tokens)
# print(make_feature(doc_tokens, vocab))

# 퀴즈
# 영화 리뷰 코퍼스로 피처를 만들어서 train_set, test_set으로 나눈 다음에
# train_set으로 학습하고 test_set에 대해 정확도를 구하세요
ids = nltk.corpus.movie_reviews.fileids()
random.shuffle(ids)

data = []
for i, name in enumerate(ids):
    doc_tokens = nltk.corpus.movie_reviews.words(name)

    features = make_feature(doc_tokens, vocab)
    target = 'neg' if 'neg' in name else 'pos'

    data.append((features, target))

    if i % 10 == 0:
        print(i)

train_set, test_set = data[:1600], data[1600:]

clf = nltk.NaiveBayesClassifier.train(train_set)
print('acc :', nltk.classify.accuracy(clf, train_set))
print('acc :', nltk.classify.accuracy(clf, test_set))

feature_set = [f for f, _ in test_set]
test_target = [t for _, t in test_set]

p = clf.classify_many(feature_set)
print(p[:5])
print(test_target[:5])
print('acc :', np.mean(np.array(p) == test_target))
