# 2_4_hamlet.py

# 퀴즈: 세익스피어의 햄릿 코퍼스에 등장하는 주인공들의 출현 빈도를
# 막대 그래프로 그려보세요

import nltk
import collections
import matplotlib
from matplotlib import pyplot as plt

txt = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')


characters = nltk.RegexpTokenizer(r'\s\s([A-Z][a-z]+)\.').tokenize(txt)
characters = list(set(characters))
print(characters)

tokens = nltk.RegexpTokenizer(r'\w+').tokenize(txt)

freq = collections.Counter(tokens)

# print(freq)

freq_characters = [(freq[i], i) for i in characters if i in freq]
print(freq_characters)
freq_characters.sort( key=lambda x: x[0], reverse=True)
print(freq_characters)

character_names = [i[1] for i in freq_characters]
character_frequencies = [i[0] for i in freq_characters]

# 막대 그래프 그리기
plt.bar(character_names, character_frequencies, width=0.5, align='center')
plt.xlabel('Characters')
plt.ylabel('Frequency')
plt.title('Character Frequency in Hamlet')
plt.xticks(rotation=90)  # 이름이 겹치지 않게 90도 회전
plt.legend()
plt.tight_layout()  # 레이아웃 조정
plt.show()

