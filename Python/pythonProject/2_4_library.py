# 2_4_library.py

import re
import requests
#
# url = "https://www.naver.com/" #네이버 서버에서 데이터를 받아와서 결과 값을 출력
# response = requests.get(url)
# print(response) # <Response [200]>: HTTPS 상태코드 : 성공
# print(response.text) # 페이즈 소스(html)를 프린트한 것이다.-> 문자열로 구성 -> 정규 표현식을 사용할 수 있다.




url = "http://211.251.214.176:8800/index.php?room_no=2" #네이버 서버에서 데이터를 받아와서 결과 값을 출력
response = requests.get(url)
# print(response) # <Response [200]>: HTTPS 상태코드 : 성공
# print(response.text) # 페이즈 소스(html)를 프린트한 것이다.-> 문자열로 구성 -> 정규 표현식을 사용할 수 있다.
# print("빈좌석 개수: ",re.findall(r'31', response.text)) # 3개 가나온다
# print("빈좌석 개수: ",re.findall(r'[0-9]+ 석</td>', response.text))
# empty = re.findall(r'([0-9]+) 석</td>', response.text) # ()의미는 여기만 리스트에 넣겠다.
# print(empty[5]

# <font style="color:green;font-size:13pt;font-family:Arial"><b>
print(response) # <Response [200]>: HTTPS 상태코드 : 성공
print(response.text) # 페이즈 소스(html)를 프린트한 것이다.-> 문자열로 구성 -> 정규 표현식을 사용할 수 있다.
print("빈좌석 번호 :", end = '')
empty = re.findall(r'<font style="color:green;font-size:13pt;font-family:Arial"><b>([0-9]+)', response.text) # ()의미는 여기만 리스트에 넣겠다.

