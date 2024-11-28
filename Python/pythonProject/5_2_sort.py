# 5_2_sort.py


def twice(n):
    return 2*n

g = lambda n: n*3

print(g(5))

f = twice   #8바이트 주소 -> 함수 포인터 -> call-back 함수
print(twice(7))
print(f(8), (lambda n: n*10)(2))

a = [85, 71, 28, 103, 17]
b = sorted(a)
# print(a)
#
# a.sort(reverse=True)    # in-place
# print(a)

def last_digit(n):
    return n%10

c = sorted(a, key = lambda n: n%10)
print(c)

#퀴즈
# 문자열 리스트를 내림차순으로 정렬하세요
colors = ['red', 'blue', 'green', 'yellow', 'CYAN', "WHITE"]

d = sorted(colors, reverse=True, key = lambda n: n.lower())
print(d)

e = sorted(colors, reverse=True, key = lambda n: len(n))
print(e)
print(e)

