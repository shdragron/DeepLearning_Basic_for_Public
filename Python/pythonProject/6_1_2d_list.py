# 6_1_2d_list.py

import random
random.seed(23)
# 퀴즈: 100보다 작은 난수가  10개 들어 있는 리스트를 만드세요.

a = [random.randrange(100) for i in range(10)]
b = [random.randrange(100) for p in range(5)]
c = [random.randrange(100) for v in range(7)]

# print(a)

# 퀴즈: 2차원 리스트를 출력하는데 반복문을 2개 쓰세요.
print(a)
print(b)
print(c)
print()

d =[a,b,c]

for i in range(len(d)):
    for j in range(len(d[i])):
            print(d[i][j], end=" ")

print()

# 퀴즈: 2차원 리스트에 들어 있는 int 갯수를 알려 주세요.
length = 0
for i in range(len(d)):
    for j in range(len(d[i])):
            length += 1
# print(length)

# print(sum(len(i) for i in d))

# 퀴즈: 2치원 리스트에서 가장 큰 숫자를 찾아보세요.

# print(max(i) for i in d)
# print(max(max(i) for i in d))

# 퀴즈: 2차원 리스트에서 홀수 합계를 구하세요.

print()

odd = 0
for i in range(len(d)):
    for j in range(len(d[i])):
        if d[i][j] % 2 == 1:
            odd += d[i][j]
print(odd)


# print(sum(i) for i in d if i % 2 == 1  )
print(sum([ j for i in d for j in i if j % 2]))
