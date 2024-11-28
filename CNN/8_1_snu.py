# 8_1_snu.py

# 퀴즈: 서울대 식단표를 웹크롤링해서 깔끔하게 보여주세요.
import requests
import certifi
import re


url = "https://snuco.snu.ac.kr/foodmenu/"

response = requests.get(url, verify=False)
print(response.text)
Food = re.findall('<td class="title">.+?<td class="dinner">', response.text, re.DOTALL)
print(len(Food))
# print(Food[0], Food[10])

for foo in Food:
    cafe = re.findall(r'<td class="title">\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t(.+?)\t\t\t\t\t\t\t\t\t\t\t\t\t</td>\r\n\t\t\t\t\t\t<td class="breakfast">',foo, re.DOTALL)
    breakfast = re.findall(r'<td class="breakfast">(.+?)<br />.+<td class="lunch">', foo,re.DOTALL)
    lunch = re.findall(r'<td class="lunch">(.+)<td class="dinner">', foo,re.DOTALL)
    dinner = re.findall(r'<td class="dinner">(.+)<td class="title">', foo,re.DOTALL)


    print(cafe, dinner)

    # city = re.findall(r'<city>(.+)</city>', loc)
    # print(prov[0], city[0])
    #
    # data = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
    # # print(len(data))
    #
    # for datum in data:
    #     mode = re.findall(r'<mode>(.+)</mode>', datum)
    #     tmEf = re.findall(r'<tmEf>(.+)</tmEf>', datum)
    #     wf = re.findall(r'<wf>(.+)</wf>', datum)
    #     tmn = re.findall(r'<tmn>(.+)</tmn>', datum)
    #     tmx = re.findall(r'<tmx>(.+)</tmx>', datum)
    #     rnSt = re.findall(r'<rnSt>(.+)</rnSt>', datum)
    #
    #     # print(prov[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0])
    #     print(prov[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0],
    #           sep=',', file=f)

















