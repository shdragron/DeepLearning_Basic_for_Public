# 5_1_for_for.py

# 퀴즈: 아래와 같이 출력하세요.
# *****
# *****
# *****
# *****
# def show_stars(n):
#     for j in range(n):
#         print("*", end= "")
#     print()
#
# for i in range( 0, 4 ):
#     show_stars(5)

# 퀴즈: 아래처럼 출력하세요.

# 1번
# *
# **
# ***
# ****

def p_1():
    for i in range(4):
        for j in range(i+1):
            print("*", end="")
        print()



# 2번
# ****
# ***
# **
# *
def p_2():
    tmp = 4
    for i in range(4):
        print("*" * tmp, end="")
        for j in range(i+1):
            print(" ", end="")
            tmp = 3 - j
        print()
# 3번

#    *
#   **
#  ***
# ****

def p_3():
    for i in range(4):
        print(" "*(4-i),end="")
        for j in range(i):
            print("*", end="")
        print()
    print("*"*4)

# 4번
# ****
#  ***
#   **
#    *

def p_4():
    for i in range(4):
        print("*"*(4-i),end="")
        for j in range(i):
            print(" ", end="")
        print()
    print(" "*4)

# 행렬 계산법

#   0123
# 0 *---
# 1 **--
# 2 ***-
# 3 ****

def p_5():
    for i in range(4):
        for j in range(4):
            if i + j <= 3:
                print("*", end = '')
            else:
                print(" ", end = '')
        print()

def p_6():
    for i in range(5):
        for j in range(5):
            if (j/2):
                print("*", end = '')
            elif(i==1 && (j==1 || j==2 ||j==)):
            else:
                print(" ", end = '')
        print()

p_1()
print()
p_2()
print()
p_3()
print()
p_4()
print()
p_5()
print()
p_5()