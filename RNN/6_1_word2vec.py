# 6_1_word2vec.py


def extract_surrounds(token_count, center, window_size):
    start = max(0, center-window_size)
    end = min(token_count, center+window_size+1)
    return [i for i in range(start, end) if i != center]


def show_dataset(tokens, window_size, is_skipgram):
    token_count = len(tokens)
    for center in range(token_count):
        surrounds = extract_surrounds(token_count, center, window_size)
        # print(surrounds)

        if is_skipgram:
            # print(center, surrounds)
            # for p in surrounds:
            #     print(center, p, tokens[center], tokens[p])
            print(*[(tokens[center], tokens[p]) for p in surrounds])
        else:
            # print(surrounds, center)
            print([tokens[p] for p in surrounds], tokens[center])


# text = 'the quick brown fox jumps over the lazy dog'
# tokens = text.split()
# print(tokens)

tokens = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

# show_dataset(tokens, window_size=2, is_skipgram=False)
show_dataset(tokens, window_size=2, is_skipgram=True)
