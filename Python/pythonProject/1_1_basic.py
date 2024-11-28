# 1_1_basic.py

#퀴즈
#hello python을 세번 출력하는 코드를 만드세요
def f_1():
    print("hello python!")
    print("hello python!")
    print("hello python!")
def f_2():
    for i in range(0,3):
        print("hello python!")
def f_3():
    a = 1
    while(a < 4):
        print("hello python!")
        a = a + 1
def f_4():
    print("hello world!","Hello Python")

#퀴즈 hello 주변에 작은 따옴표와 큰 따옴표를 출력하세요.

f_1()
f_2()
f_3()
f_4()

print("'''")
print()

#     int  float  bool  str
print(12, 3.14, True, "coffee")
print(type(True))

print(int(3.14))

# a와 b의 값을 교환하세요

a = 7
b = 3
tmp = 0;
tmp = a;
a = b
b = tmp
print(a, b)
print()

c = 7
d = 3

c, d = d, c
print(c,d)

