# 3_2_dict.py
b = [1, 3, 5, 1, 2, 3, 2, 3, 5]
# a = {}
# 퀴즈: 리스트에서 중복된 값들을 모두 삭제하세요.
a = set(b)
print(a)
print(list(a))

# 한영 사전: 한글 단어를 찾으면 영어 설명이 나옴
# 한글단어: Key
# 영어단어: value
d = { 'name': 'kim', 'age': 21}
print(d)
print(d['name'], d['age'])
d['hobby'] = 'climbing'
print(d)
d['hobby'] = 'paragliding'
print(d)
print(d.keys())
print(d.values())
print(d.items())

exit()
# 퀴즈
# 딕셔너리의 items 함수를 for문에 적용하세요.

for k, v in d.items():
	print(k, v)