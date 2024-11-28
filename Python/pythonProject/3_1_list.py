# 3_1_list.py

def show_me(money = 100): # defalt parameter
    print(money);

def key_function(a, b, c):
    print(a, b, c)

def test_func(a=0,b=0,c=0):
    print(a,b,c)
def changable(*args):
    print(args)


a = [1,'abc',4,2]
print(a)
print(type(a))

s = "hello"
print(len(s))
print(a[-1])

# 0,1,2,3,4 -> 0 4 1

for i in range(0,4,1):  # 시작, 종류, 증감
    print(i, end = ' ')
print()

for i in range(0,4):    # 시작, 종료, 1
    print(i, end = ' ')
print()

for i in range(4):  # 0, 시작, 1
    print(i, end = ' ')
print()

show_me() # None 반환
# help(print)
key_function(1, 2, 3)   # positional argument
key_function( a=1, b=2, c=4)    # keyword argument
# key_function( a=1, 2, c=4)

# 퀴즈 함수를 호출하는 5가지 방법 (다형성)
# test_func(1, 2, 3)
# test_func(a=1, b=2, c=3)
# test_func(1, b=2, c=3)
# test_func(1, 2, c=3)
# test_func(1, 2, 3)
# test_func()
# test_func(c=3)
# print()
# changable()
# changable(12)
# changable(12,'c')
# changable(12,'c',3.14)
print()

for i in range(100):
    print('-', end = '')
# a, b = 3, 7
# c = 3, 7
#
# print(a, b, c)

t = [1,3,5]
print(t)
print(*t)
