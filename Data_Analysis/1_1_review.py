# 1_1_review.py

import random

random.seed(23)
# 퀴즈
# 아래처럼 출력하세요
# 0
# 0 1
# 0 1 2
# 0 1 2 3

for i in range(4):
    for j in range(4):
        if (i==j):
            print(j, end = " ")
        elif(i>j):
            print(j, end = " ")
    print()

# 퀴즈
# 0에서 4까지 난수 10개로 이루
idx = []
for i in range(10):
    idx.append(random.randint(0,4))
print(idx)

# 리스트에 들어 있는 값의 빈도를 구하세요
tmp = [9]
j0 = 0
j1 = 0
j2 = 0
j3 = 0
for i in idx:
    for t in tmp:
        if i == t:
            if i == 0:
                j0 += 1
                break
            elif i == 1:
                j1 += 1
                break
            elif i == 2:
                j2 += 1
                break
            elif i == 3:
                j3 += 1
                break
        else:
            tmp.append(i)
print("빈도 0, 1, 2, 3: ", j0, j1, j2, j3)

b0 = [random.randrange(100) for _ in range(10)]
b1 = [random.randrange(100) for _ in range(10)]
b2 = [random.randrange(100) for _ in range(10)]
print(b0, b1, b2)

c = [b0, b1, b2]
print(c)
# 퀴즈
# 2차원 리스트를 1차원 리스트로 변환하세요.

# tmp = [i for i in b0 + b1 + b2]
#
# print(tmp)

# 퀴즈

print([j for i in c for j in i if j % 2 ])
print([[j for j in i if j % 2 ] for i in c])

# 퀴즈:
# 1~10000 사이의 정수에 포함된 8의 갯수를 구하세요
# 808: 2개

count = 0
for i in range(10000):
    for j in str(i):
        if j == '8':
            count += 1
print(count)


print(sum(1 for i in range(10000) for j in str(i) if j == '8'))

print(sum(str(i).count('8') for i in range(10000)))



