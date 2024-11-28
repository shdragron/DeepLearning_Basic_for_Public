# 2_3_gensim.py

import gensim
import nltk


def gensim_basic():
    text = ['나는 너를 사랑해','나는 너를 미워해']

    tokens = [t.split() for t in text]
    print(tokens)

    embeddings = gensim.models.Word2Vec(tokens, min_count=1, vector_size=2)

    print(embeddings) # Word2Vec<vocab=4, vector_size=2, alpha=0.025>
    print(embeddings.wv) # KeyedVectors<vector_size=2, 4 keys>
    print(embeddings.wv.vectors)
    # [[-0.02681136  0.01182157]
    #  [ 0.25516748  0.45046365]
    #  [-0.4651475  -0.35584044]
    #  [ 0.32294363  0.4486494 ]]
    print(embeddings.wv.key_to_index)
    # {'너를': 0, '나는': 1, '미워해': 2, '사랑해': 3}
    print(embeddings.wv.index_to_key)
    # ['너를', '나는', '미워해', '사랑해']
    # ONEHOT 인코딩보다 이것을 사용하는 이유는 연관성을 주기 때문에
    print(embeddings.wv['나는'])
    # [0.25516748 0.45046365]

def gensim_similarity():
    sents = nltk.corpus.movie_reviews.sents()
    print(sents)
    print(len(sents))

    model = gensim.models.Word2Vec(sents)
    # 코사인 유사도 -> 내적
    # 비슷(1.0), 다르다(-1.0), 상광없음(0)
    print(model.wv.similarity('kind','villain'))    # 0.42894837
    print(model.wv.similarity('man','woman'))       # 0.87965655
    print(model.wv.similarity('sky','book'))        # 0.45673543
    print(model.wv.similarity('hello','toilet'))    # 0.75576526
    print(model.wv.similarity('mom','theory'))      # 0.63714683

    print(model.wv.most_similar('theory'))

# gensim_basic()
gensim_similarity()





