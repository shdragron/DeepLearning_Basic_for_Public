# 5_4_library.py
import re
import requests


# 퀴즈
# 성남시 해오름도서관에서 책 제목 검색한 결과를 파싱하세요
# 출력: 제목, 저자, 출판사, 발행연도
payload = {
    'searchType': 'SIMPLE',
    'searchCategory': 'BOOK',
    'searchKey': 'ALL',
    'searchLibraryArr': 'MH',
    'topSearchType': 'BOOK',
    'searchKeyword': '고래'
}

url = 'https://www.snlib.go.kr/hor/plusSearchResultList.do'
response = requests.post(url, params=payload)
# print(response)
# print(response.text)

result = re.findall(r'<ul class="resultList imageType">(.+?)</ul>', response.text, re.DOTALL)
# print(len(result))
# print(result[0])

li = re.findall(r'<li>(.+?)</li', result[0], re.DOTALL)
# print(len(li))
# print(li[0])

for item in li:
    # title = re.findall(r'<a.+?>(.+?)</a>', item, re.DOTALL)
    title = re.findall(r'title="(.+?) 선택"', item, re.DOTALL)
    writer = re.findall(r'<span>저자 : (.+?)지음', item)
    year = re.findall(r'<span>발행연도: (.+?)</span>', item)
    publisher = re.findall(r'<span>발행자: (.+?)</span>', item)
    publisher = publisher[0].replace('<span class="searchKwd themeFC">', '')
    print(title)
    print(writer)
    print(year)
    print(publisher)
    print('-' * 30)

# <a href="#link" onclick="javascript:fnSearchResultDetail(1094852123,1094852125,'BO'); return false;">
# 								칭찬은 <span class="searchKwd themeFC">고래</span>도 춤추게 한다
# 							</a>
# 							<span>저자 : 켄 블랜차드 [외] 지음  ; 조천제 옮김</span>
# 								<span>발행연도: 2014</span>
# 							<span>발행자: 21세기북스</span>