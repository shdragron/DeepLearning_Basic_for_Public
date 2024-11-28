import re

# 퀴즈: 기상청 데이터를 json에서 했던 것처럼 예쁘게 출력하세요.(정규 표현식을 이용해서)
json_text = "[{'code': '11', 'value': '서울특별시'}, {'code': '26', 'value': '부산광역시'}, {'code': '27', 'value': '대구광역시'}, {'code': '28', 'value': '인천광역시'}, {'code': '29', 'value': '광주광역시'}, {'code': '30', 'value': '대전광역시'}, {'code': '31', 'value': '울산광역시'}, {'code': '41', 'value': '경기도'}, {'code': '42', 'value': '강원도'}, {'code': '43', 'value': '충청북도'}, {'code': '44', 'value': '충청남도'}, {'code': '45', 'value': '전라북도'}, {'code': '46', 'value': '전라남도'}, {'code': '47', 'value': '경상북도'}, {'code': '48', 'value': '경상남도'}, {'code': '50', 'value': '제주특별자치도'}]"

empty = re.findall(r"{'code': '([0-9]+)', 'value': '([가-힣]+)'}",json_text)  # ()의미는 여기만 리스트에 넣겠다.
print(empty)
area = []
codes = []
for i in range(len(empty)):
    print(empty[i][0], empty[i][1])
    area.append(empty[i][1])
    codes.append(empty[i][0])

print(area)
print(codes)
print()

for code, area in zip(codes, area):
    print(code, area)