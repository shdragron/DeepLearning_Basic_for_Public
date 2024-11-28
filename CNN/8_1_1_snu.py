# 8_1_snu.py
import re
import requests


# 퀴즈
# 서울대 식단표를 웹크롤링해서 깔끔하게 보여주세요
def show_food_menu(year,month,day):

    # https://snuco.snu.ac.kr/foodmenu/?date={}-{}-{}

    url = 'https://snuco.snu.ac.kr/foodmenu/?date={}-{:02}-{}'.format(year,month,day)
    response = requests.get(url, verify=False)
    # print(response)
    # print(response.text)

    menu_table = re.findall(r'<table class="menu-table">(.+?)</table>',
                            response.text, re.DOTALL)
    # print(menu_table[0])

    tbody = re.findall(r'<tbody>(.+?)</tbody>', menu_table[0], re.DOTALL)
    # print(tbody[0])

    trs = re.findall(r'<tr>(.+?)</tr>', tbody[0], re.DOTALL)
    # print(len(trs))

    for tr in trs:
        # print(tr)
        tds = re.findall(r'<td.*?>(.*?)</td>', tr, re.DOTALL)
        tds = [td.strip() for td in tds]
        tds = [td.replace('<br />', '') for td in tds]
        tds = [td.replace('&nbsp;', '') for td in tds]
        tds = [td.replace('\n\r', '') for td in tds]
        tds = [td.replace('&amp;', '&') for td in tds]
        tds = [td.replace('&lt;', '<') for td in tds]
        tds = [td.replace('&gt;', '>') for td in tds]

        # print(len(tds), tds[0])

        print('*', '=' * 30, '*')
        print(tds[0])
        print('-- 조식')
        print(tds[1] if tds[3] else '운영 안함')
        print('-- 중식')
        print(tds[2] if tds[3] else '운영 안함')
        print('-- 석식')
        print(tds[3] if tds[3] else '운영 안함')
        print()

show_food_menu(2024,8,30)



