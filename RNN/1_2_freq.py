# 1_2_freq.py
import collections

import nltk
import re
import operator

# print(nltk.corpus.webtext.fileids())

# 퀴즈: wine.txt. 파일을 가져와서 토큰으로 나눠주세요

wine = nltk.corpus.webtext.raw('wine.txt')
# print(wine)

# tokenzie
# tokens = re.findall(r'\w+', wine)
wine = nltk.corpus.webtext.raw('wine.txt')
# print(wine)

wine = wine.lower()
tokens = nltk.RegexpTokenizer(r'\w+').tokenize(wine)
# print(tokens)
# 한글자 단어 삭제 'a', 't'
tokens = [str for str in tokens if len(str) > 1]
# print(len(tokens))
# print(tokens)

# 삭제 'and', 'but', 'the'
tokens = [str for str in tokens if str not in ['the','and','but']]
# print(len(tokens))
# print(tokens)

# -> 불용어
tokens = nltk.RegexpTokenizer(r'\w+').tokenize(wine)

# nltk.download('stopwords')
stop_word = nltk.corpus.stopwords.words('english')

tokens = [str for str in tokens if not str in stop_word]
print(len(tokens))

# 퀴즈
# 토큰 별 빈도를 구해서, 가장 많이 나타난 토큰 10개를 알려주세요.

freq = {}
for key in tokens:
    # if key not in freq:
    #     freq[key] = 0
    # freq[key] += 1

    freq[key] = freq.get(key, 0) + 1
# print(freq)

counts = sorted([(freq[key], key) for key in freq], reverse=True)
# counts = list(freq.items())
print(counts[:10])


counts = list(freq.items())
counts.sort(key=lambda x: x[1], reverse=True)
# counts.sort(key=operator.itemgetter(1), reverse=True)
print(counts[:10])

freq2 = collections.Counter(tokens)
print(freq2)

freq3 = nltk.FreqDist(tokens)
print(freq3.most_common(10))








