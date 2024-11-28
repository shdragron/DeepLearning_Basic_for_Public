# 4_1_food.py
import requests
import re


# 퀴즈
# 한국폴리텍대학교 신기술교육원의 식단 데이터를 파싱하세요
payload = {
    'day': '20240729',
}

url = "https://www.kopo.ac.kr/int/content.do?menu=2520"
response = requests.post(url, data=payload)
# print(response)
# print(response.text)
# print(response.content)

text = response.text.replace('\r', '')
print(text)

tbody = re.findall(r'<tbody>(.+?)</tbody>', text, re.DOTALL)
# print(tbody)
# print(len(tbody))
# print(tbody[0])

tr = re.findall(r'<tr>(.+?)</tr>', tbody[0], re.DOTALL)
# print(len(tr))
# print(*tr, sep='\n')

weekday = ['월', '화', '수', '목', '금', '토', '일']
for item, day_text in zip(tr, weekday):
    print('--------', day_text)

    # print(item)
    td = re.findall(r'<td>(.+?)</td>', item, re.DOTALL)
    # print(*td, sep='\n')

    # foods = []
    meal = ['조식', '중식', '석식']
    for i, (cell, meal_text) in enumerate(zip(td[1:], meal)):
        cell = re.sub(r'<.?span.*?>', '', cell)
        # cell = cell.replace('<span>', '')
        # cell = cell.replace('</span>', '')
        # cell = cell.replace('<span class="kcal">', '')

        print('{} : {}'.format(meal_text, cell.strip()))
    #     foods.append(cell.strip())
    # print(*foods, sep='\n')
    # print(foods)
    # print('--------------')
