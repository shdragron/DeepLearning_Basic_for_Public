# 4_1_slicing.py
a = list('abcdefg')
print(a)
#
# # print(a[-1])
# # print(a[0:4])
# # print(a[4:7])
#
# print(a[::2]) # 시작 종료 몇 칸씩 -> 시작 종료 안쓰면 None이 들어감.
# print(a[1::2]) # 시작 종료 몇 칸씩
# print(a[7:None:-1])
# print(a[::-1])

# 1)
for i in range(len(a)):
    print(a[i], end = ' ')
print()
# 2) 객체 넘버링 -> (인덱스값,데이터값) 튜플
for i, v in enumerate(a):
    print(i, v, end = ' ')
print()

# 3) 역변환
for i in reversed(a):
    print(i, end = ' ')
print()
