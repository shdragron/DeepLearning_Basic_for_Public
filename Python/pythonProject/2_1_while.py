# 2_1_while.py

#퀴즈
#아래 숫자들의 규칙을 숫자 3개로 표현하세요.
#01234  0,5,1
#02468  0,10,2
#43210  4,-1,-1

#1
# i = 0
# while i < 5:
#     print(i, end = '')
#     i += 1
#2
# i = 0
# while i < 10:
#     print(i, end = '')
#     i += 2
#3
# i = 4
# while i > -1:
#     print(i, end = '')
#     i += -1
#퀴즈
# 0~9까지의 정수를 거꾸로 출력하세요.
# i = 9
# while i > -1:
#     print(i, end = ' ')
#     i += -1
# 0~24 사이의 정수를 출력허세요. (한줄의 5개씩) (2가지)
# 1번 방식
i = 0
while i < 25:
    if (i+1) % 5 == 0:
        print(i)
        i += 1
    else:
        print(i ,end = ' ')
        i += 1
print()
# 2번 방식
i = 0
tmp = 0
while i < 25:
    tmp += 1
    print(i ,end = ' ')
    i += 1
    if tmp % 5 ==  0:
        print()
