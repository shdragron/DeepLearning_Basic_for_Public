# 5_4_tokenizing.py
import tensorflow.keras as keras
import numpy as np

motto = "If you want to build a ship, don't drum up people to collect wood " \
        "and don't assign them tasks and work, but rather teach them " \
        "to long for the endless immensity of the sea."

tokenizer = keras.preprocessing.text.Tokenizer(num_words=10)
tokenizer.fit_on_texts(motto.split())
print(tokenizer)
print(tokenizer.index_word)
# {1: 'to', 2: "don't", 3: 'and', 4: 'them', 5: 'the', 6: 'if', 7: 'you',
#  8: 'want', 9: 'build', 10: 'a', 11: 'ship', 12: 'drum', 13: 'up', 14: 'people',
#  15: 'collect', 16: 'wood', 17: 'assign', 18: 'tasks', 19: 'work', 20: 'but',
#  21: 'rather', 22: 'teach', 23: 'long', 24: 'for', 25: 'endless',
#  26: 'immensity', 27: 'of', 28: 'sea'}
print(tokenizer.word_index)

print(tokenizer.index_word[9])
print(tokenizer.index_word[10])
print(tokenizer.index_word[11])
print()

print(tokenizer.texts_to_sequences(['build', 'a', 'ship']))
print(tokenizer.texts_to_sequences([['build', 'a', 'ship']]))
print(tokenizer.texts_to_sequences([['build'], ['a'], ['ship']]))
print()

seq = [[7, 8, 9, 1, 2], [13, 14, 5], [6, 7]]
print(tokenizer.sequences_to_texts(seq))
print()

print(keras.preprocessing.sequence.pad_sequences(seq))

print(keras.preprocessing.sequence.pad_sequences(seq, padding='pre'))
print(keras.preprocessing.sequence.pad_sequences(seq, padding='post'))
print()

print(keras.preprocessing.sequence.pad_sequences(seq, maxlen=7))
print(keras.preprocessing.sequence.pad_sequences(seq, maxlen=3))
print()

print(keras.preprocessing.sequence.pad_sequences(seq, maxlen=3, truncating='pre'))
print(keras.preprocessing.sequence.pad_sequences(seq, maxlen=3, truncating='post'))
print()

# eye = np.eye(5, dtype=np.int32)
eye = np.identity(5, dtype=np.int32)
print(eye)
print()

print(eye[2])
print(eye[[2, 3]])
print(eye[[[2, 3], [1, 2]]])

