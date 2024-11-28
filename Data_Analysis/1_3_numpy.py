# 1_3_numpy.py
import numpy as np
import random

a = np.arange(12)
b = np.arange(0,10)
c = np.arange(0,12,2)

print(a)
print(b)
print(c)
print(a.shape,a.dtype, a.size, a.ndim) # 1. 차원 확인, 2. 데이터 형태 타입은 8바이트가 기본이다. 3. 사이즈, 4.
b =np.array([1,2,3,4,5,6,7,8,9,10])

a = np.arange(12)
print(a + 1) # broadcasting
print( a + a) # vector 연산
print(np.sin(a)) # univsrsal function

print('-'*50)

# b = a.reshape(3,4)
# b = a.reshape(3,-1)
b = a.reshape(-1,4)

print(b)
a[0] = 99
print(a)
print(b)

# 퀴즈: 2차원 배열을 1차원으로 바꿔주세요. (3가지 코드)

print(b.reshape(12))
print(b.reshape(b.size))
print(b.reshape(-1))

print("-"*50)

a[0] = 0

print(b + 1)
print(b+b)
print(np.sin(b))

print("-"*50)

a0 = np.arange(6)
a1 = np.arange(6).reshape(2,3)
a2 = np.arange(3)
a3 = np.arange(3).reshape(1,3)
a4 = np.arange(6).reshape(3,2)
a5 = np.arange(6)
a6 = np.arange(6).reshape(-1,1)
print(a3, '!')
print(a6, '!')
print(a5, '!')
# print(a0 + a1) # 차원이 다르고 크기가 열크기가 아니면 안된다.

print(a1 + a2) # 차원이 다르지만 작은 열크기와 크기가 같으므로 행에다 두번 더한다.

# print(a1 + a4) # 차원이 다르고 크기가 열크기가 아니면 안된다.
# reshape(열, 행)
print(a5 + a6)

print(np.array([random.randrange(10) for _ in range(10)]))
print(np.random.randint(0,10,size=10))

print(c[c > 5])


