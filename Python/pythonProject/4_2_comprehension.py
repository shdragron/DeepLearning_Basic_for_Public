# 4_2_comprehension.py

import random
random.seed(17)
# 컴프리헨션: (리스트를 만드는)한 줄짜리 반복문

# 원래
a = []
for i in range(10):
    a.append(i)
print(a)

# 컴프리헨션 사용

b = [i for i in range(10)] #리스트안에 들어갈것 + 반복문
print(b)

# 퀴즈: 100보다 작은 난수 10개로 리스트를 만드세요.

c = [random.randint(0,100) for _ in range(10)]
print(c)

d = [random.randrange(0,100,1) for _ in range(10)]
print(d)

# 퀴즈 리스트에서 홀수만 뽑아서 새로운 리스트를 만드세요.

e = [i for i in d[1::2]]
print(e)
