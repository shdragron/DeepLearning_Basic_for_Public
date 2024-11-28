# 2_3_re.py

import re  # regular expression
# """ 여기에서는 계행 문자가 안보이게 포함되어 있다.
db = """3412    [Bob] 123
3834  Jonny 333
1248   Kate 634
1423   Tony 567
2567  Peter 435
3567  Alice 535
1548  Kerry 534"""

# print(db)
print(re.findall(r'\d', db))    # 숫자 찾기, 결과 값은 LIST이고 인자는 문자열이다.
print(re.findall(r'[0-9]', db))
print(re.findall(r'[0-9]+', db))
print(re.findall(r'[0-9]{4,}', db))
# r: raw string

# 퀴즈
# 직원 이름을 찾아 주세요.
print(re.findall(r'[A-z]+', db)) # 아스키코드 살펴보기 wrong
print(re.findall(r'[A-Za-z]+', db)) # 데이터에 대한 이해가 필요! Just
print(re.findall(r'[A-Z][a-z]+', db)) #대문자 앞에 good

# 퀴즈
# T로 시작하는 이름만 찾아보세요
print(re.findall(r'[T][a-z]+',db))
print(re.findall(r'[^Ta-z][a-z]+',db))

