#1_2_operator

#연산 ; 산술, 관계, 논리, 비트

# 산술 연산: + - * / % ** //
# print(2**3)
# print(8//3) #정수 나눗셈
# print(8/3) #실수 나눗셈

#퀴즈 0~999 사이의 양수의 자릿수를 알려주세요.
# 1
# a = 432
#
# b  = a//100
# print(b)
# c = a//10 - b*10
# print(c)
# d = a - (a//10)*10
# print(d)
#
# print(b + c + d)
#
# # 2a
# e = 134
# s = 0
#
# while e > 0:
#     s += e%10
#     e //= 10
# print(s)
#
# i = input('정수입력:')
# print(i)


#퀴즈
#입력한 양수와 크거나 같은 첫번째 5의 배수를 구하시오
# print(100%5)
# a = int(input('정수: '))
# print(((a + 4) // 5) * 5)

# 관계 : > >= <= < == !=
# 퀴즈
# age가 10대인지 알려주세요.
# age = int(input("나이: "))
# print(10<=age<=19)

# 논리: and or not

print(True and False)

# int float boolean str

print("hello" + "python"*3, end = '')
print("----"*10)