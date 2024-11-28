#2_5_open_hangul.py

# 퀴즈 오픈 한글_ 한글->알파벳 메뉴를 사용해서
# 한글을 영어로 변환 하는 함수를 만드세요.

import re
import requests

han = input("")


def h2e(han):
    url = "http://openhangul.com/nlp_ko2en?q=" + han  # 오픈 한글과 검색 결과 복사(url)
    response = requests.get(url)
    # print(response) # <Response [200]>: HTTPS 상태코드 : 성공
    # print(response.text) # 페이즈 소스(html)를 프린트한 것이다.-> 문자열로 구성 -> 정규 표현식을 사용할 수 있다.
    print("한글 -> 알파벳 :", end='')
    empty = re.findall(r'<img src="images/cursor.gif"><br>([.]+))', response.text)  # ()의미는 여기만 리스트에 넣겠다.
    print(empty)


h2e(han)