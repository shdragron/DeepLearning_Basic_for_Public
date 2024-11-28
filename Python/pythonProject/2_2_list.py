#2_2_list.py
#
# a = [1,2,3]
# print(a)
# a.append(9)
# a.insert(1,5)
#
# i = 0
# while i < 3:
#     print(i, a[i])
#     i += 1
# print(len(a))

# 퀴즈 리스트를 거꾸로 출력하세요.
#
# i = len(a) - 1
# a.pop(len(a)-1)
# while i > -1:
#     print(i, a[i])
#     i -= 1
# 퀴즈 리스트에서 마지막 요소를 삭제하세요
# 퀴즈 0~99사이의 난수를 10개 출력하세요.
import random

random.seed(23)
a = []
i = 0
print(random.randrange(0,100))
while i < 10:
    a.append(random.randrange(0,100))
    i += 1
print(a)
# 퀴즈: 리스트에 들어 있는 홀수 합계와 짝수 합계를 구하시

i2 = 0
odd = 0
even = 0

while i2 < 10:
    if a[i2] % 2 == 0:
        odd += a[i2]
        i2 += 1
    else:
        even += a[i2]
        i2 += 1

print (odd, even)



