
#매개변수 있고 반환값있고
#
# def hungry(a):
#     b = a
#     n = a * 3
#     return n
# print(hungry(5))
#
# def df():
#     pass
# a , b= int(input("Num: ")), int(input("Num: "))
#
# def big(c,d):
#     if c > d:
#         return c
#     else:
#         return d
#
# print(big(a,b))

# 퀴즈
#4개의 정수로 부터  큰값을 찾는 함수를 만드세요

a , b , c, d= int(input("Num: ")), int(input("Num: ")),int(input("Num: ")),int(input("Num: "))

1
# def big(e,f,g,h):
#     if e >= f and e >= g and e >= h:
#         return e
#     elif f >= g and f >= h:
#         return f
#     elif g >= h and g >= f:
#         return g
#     elif h >= f:
#         return h
# def big(e,f,g,h):
#     if e <= 0 or f <= 0 or  g <= 0 or h <= 0:
#         if e+f+g+h <= 0:
#             return (-1)*((e**2+f**2+g**2+h**2/4)+((e+f+g+h)/4))
#         else:
#             return ((root(e**2+f**2+g**2+h**2/16))+((e+f+g+h)/4))
#     else: return ((e**2+f**2+g**2+h**2/4)+((e+f+g+h)/4))

def big(e,f,g,h):
    if e < f : e = f
    elif e < g : e = g
    elif e < h : e = h
    else: e = h
    return e

def max_2(a,b):
    return a if a>b else b


def max_3(a,b,c,d):
    a = max_2(max_2(a, max_2(b,c)),d)
    b = max_2(max_2(a,b),max_2(c,d))
    return a, b


print(max_3(a,b,c,d))


