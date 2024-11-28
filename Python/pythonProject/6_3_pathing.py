# 6_3_pathing.py
import requests
import re
# payload = {
# "S_PAGENUMBER" : "1",
# "S_MB_CD" : "W0726200"
# }
# url = "https://www.komca.or.kr/srch2/srch_01_popup_mem_right.jsp"
# # 해당 페이지를 가져오지 못했다.
# # 해당 사이트 개발자 도구 네트워크에서 헤더 -> request url: 이주소, 요청 메서드 두 가지 1) get, 2) Post 방식
# # get: 주소창을 사용 -> 단점: 1. 보안 취약 2. 길이 제한
# # POST: 백그라운드 전송 -> 암호화, 대용량 데이터 전송, 폼데이터(양식 데이터 - 웹 버튼)
# # payload
#
# # <th scope="col">저작물명</th>
# # <th scope="col">가수명</th>
# # <th scope="col">작사</th>
# # <th scope="col">작곡</th>
# # <th scope="col">편곡</th>
# response = requests.post(url, data=payload)
# # print(response)
#
# tbody = re.findall(r"<tbody>(.+?)</tbody>", response.text, re.DOTALL)
# # print(len(tbody))
# # print(tbody[1])
#
# tr = re.findall(r"<tr>(.+?)</tr>", tbody[1],re.DOTALL)
# print(len(tr))
# print(tr[0])

# for row in tr: # sub: 치환, findall: 찾기
#     row  = re.sub(r"<br/>",",",row)
#     # row  = re.sub(r' <img src="/images/common/control.gif" alt="" />',"", row)
#     # row  = re.sub(r' <img src="/images/common/control.gif"  alt="" />',"", row)
#     row  = re.sub(r' <img .+? />',"", row)
#
#     # row = row.replace('<br/>',',')
#     td = re.findall(r"<td>(.+?)</td>", row)
#     td[0] = td[0].strip() # 양쪽 공백 제거
#     print(td)

# 퀴즈: 앞에서 만든 코드를 함수로 만드세요.(코드, 페이지 번호)

def get_songs(code,page):

    payload = {
        "S_PAGENUMBER": page,
        "S_MB_CD": code
    }
    url = "https://www.komca.or.kr/srch2/srch_01_popup_mem_right.jsp"
    response = requests.post(url, data=payload)
    # print(response)

    tbody = re.findall(r"<tbody>(.+?)</tbody>", response.text, re.DOTALL)

    tr = re.findall(r"<tr>(.+?)</tr>", tbody[1], re.DOTALL)
    if len(tr) == 0:
        return False
    for row in tr:  # sub: 치환, findall: 찾기
        row = re.sub(r"<br/>", ",", row)
        # row  = re.sub(r' <img src="/images/common/control.gif" alt="" />',"", row)
        # row  = re.sub(r' <img src="/images/common/control.gif"  alt="" />',"", row)
        row = re.sub(r' <img .+? />', "", row)

        # row = row.replace('<br/>',',')
        td = re.findall(r"<td>(.+?)</td>", row)
        td[0] = td[0].strip()  # 양쪽 공백 제거
        print(td)
    return True

page = 1
while(get_songs(code="W0726200", page = page)):
    print("-"*100)
    page += 1

# 웹크롤링에서 로봇 입니까 -> 방지 -> robot.txt가 있어서 텀이 나온다. 그 텀을 주면 된다.

