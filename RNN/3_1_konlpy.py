# 3_1_konlpy.py
import konlpy
import collections


print(konlpy.corpus.kolaw.fileids())
print(konlpy.corpus.kobill.fileids())

f = konlpy.corpus.kolaw.open('constitution.txt')
print(f)
kolaw = f.read()
f.close()

tokens = kolaw.split()
print(tokens)

# pos: part of speech
tagger = konlpy.tag.Kkma()
# tagger = konlpy.tag.Hannanum()

pos = tagger.pos(kolaw)
print(pos[:10])

print('nmorphs :', len(pos))
print('nmorphs :', len(set(pos)))

print('tokens  :', len(tokens))
print('tokens  :', len(set(tokens)))


