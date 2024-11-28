# 1_1_nltk.py

import nltk
import re

def data_set():
    nltk.download('gutenberg')
    nltk.download('webtext')
    nltk.download('reuters')

    nltk.download()

def corpus():
    print(nltk.corpus.gutenberg)
    print(nltk.corpus.gutenberg.fileids())
    # ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt',
    #  'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt',
    #  'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt',
    #  'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt',
    #  'whitman-leaves.txt']
    print(nltk.corpus.gutenberg)
    c1 = nltk.corpus.gutenberg.raw('austen-emma.txt')
    print(c1)
    print(type(c1))

    # '['가 포한된다.
    c2 = nltk.corpus.gutenberg.words('austen-emma.txt')
    print(c2)
    print(type(c2))
    print(len(c2)) # 192427

    c3 = nltk.corpus.gutenberg.words()
    print(c3)
    print(len(c3)) # 2621613

def tokenize():
    emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    emma = emma[:1000]
    print(emma)
    print('-' * 100)
    print()
    print(nltk.tokenize.simple.SpaceTokenizer().tokenize(emma)) # 공백으로 토큰화
    print('-' * 100)
    # nltk.download('punkt_tab')
    sents = nltk.tokenize.sent_tokenize(emma) # 문장으로 토큰화
    print(sents)
    print(len(sents)) # 6
    print(sents[0]) # [Emma by Jane Austen 1816]
    print('-' * 100)
    # 다 동일
    # 1
    token = re.findall(r'[a-zA-Z0-9]+', emma)
    print(token)
    # 2
    t1 = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9]+').tokenize(emma)
    print(t1)
    print(len(t1))
    # 3
    t2 = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(emma)
    print(t2)
    print(len(t2))
    # 번외 - \w 의 반대
    t3 = nltk.tokenize.RegexpTokenizer(r'\W+').tokenize(emma)
    print(t3)
    print(len(t3))


# 어간 추출
# 우리: 우리는, 우리가, 우리와, 우리를, ....
def stemming():
    words = ['lives', 'dies', 'files','died']

    # Porter
    st = nltk.stem.PorterStemmer()
    print(st.stem(words[0])) # live
    stem = [st.stem(words[i]) for i in range(len(words))]
    print(stem)
    print('-'*100)
    # Lancaster
    st = nltk.stem.LancasterStemmer()
    print(st.stem(words[0]))  # live
    stem = [st.stem(words[i]) for i in range(len(words))]
    print(stem)


def grams():
    text = 'John works in Intel for vacation'

    # mini gram
    tokens = nltk.word_tokenize(text)
    print(tokens)

    #bigram
    # 1
    bi_tokens = [(a1, a2) for (a1, a2) in zip(tokens, tokens[1:])]
    print(bi_tokens)
    # 2
    bi_tokens = list(nltk.bigrams(tokens))
    print(bi_tokens)

    # quadgrams
    # 1
    n_tokens = list(nltk.ngrams(tokens, 4))
    print(n_tokens)


# datasets()
# corpus()
# tokenize()
stemming()
# grams()














