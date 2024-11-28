# 3_3_json.py
import json
import requests

a = '{"ip": "8.8.8.8"}'
b = json.loads(a)   # str -> 자료형
print(b)
print(type(b))
print(b['ip'])

c = json.dumps(b) # 자료형 -> str
print(c)
print(type(c))

dt = '''{
   "time": "03:53:25 AM",
   "milliseconds_since_epoch": 1362196405309,
   "date": "03-02-2013"
}'''

# 퀴즈: 원격에서 수신한 아래 데이터로부터 값만 출력하세요.
d = json.loads(dt)  # 유니코드 -> dictionary
print(d.values())
print(d)

for i in d:
    print(d[i])


# 퀴즈: 딕셔너리를 문자열로 변환하세요.
# desktop = 'test.json'
# with open(desktop, 'w') as f:
#     json.dump(b, f)
# with open(desktop, 'r') as f:
#     test = json.load(f, encoding='utf-8')

print("--------------------"*30)

url = "https://www.kma.go.kr/DFSROOT/POINT/DATA/top.json.txt"
response = requests.get(url) # 인코딩 실패
text = response.content.decode('utf-8')
print(type(response.content))
print()
j = json.loads(text)
print(j)
print(type(j))
print(j[0])
# 퀴즈: 기상터 데이터를 수신해서 지역 이름과 코드만 예쁘게 출력하세요.
for i in range(len(j)):
    print(j[i]["code"],j[i]["value"])

print("--------------------"*30)
# json_text = '[{'code': '11', 'value': '서울특별시'}, {'code': '26', 'value': '부산광역시'}, {'code': '27', 'value': '대구광역시'}, {'code': '28', 'value': '인천광역시'}, {'code': '29', 'value': '광주광역시'}, {'code': '30', 'value': '대전광역시'}, {'code': '31', 'value': '울산광역시'}, {'code': '41', 'value': '경기도'}, {'code': '42', 'value': '강원도'}, {'code': '43', 'value': '충청북도'}, {'code': '44', 'value': '충청남도'}, {'code': '45', 'value': '전라북도'}, {'code': '46', 'value': '전라남도'}, {'code': '47', 'value': '경상북도'}, {'code': '48', 'value': '경상남도'}, {'code': '50', 'value': '제주특별자치도'}]'

