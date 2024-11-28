# 4_5_csv.py


# 퀴즈
# weather.csv파일일 읽고 예쁘게 출력하는 함수를 만드세요
# 서울ㆍ인천ㆍ경기도::서울::A02,2024-07-24 12:00::흐리고 비::25::31::70

import csv
import re
import requests

# 내 방법
# def read():
#     f = open("data/weather_1.csv", 'r', encoding="utf-8") #'w' 초기화 후 쓰기모드
#     str1 = f.read()
#     return str1
#     f.close()


# read()
#
# i1 = re.findall(r'(.+),(.+),(.+),(.+),(.+),(.+),(.+),(.+)\n',read())
# print(*i1)
# print(type(i1))
# for a in range(len(i1)):
#     print(*i1[a],sep='::')

# 강사님 방법




def show_csv():
    f = open("data/weather_1.csv", 'r', encoding="utf-8") #'w' 초기화 후 쓰기모드
    # for line in f:
    #     print("::".join(line.strip().split(',')))
    # f.close()
    for line in csv.reader(f):
        print(line)
    print(type(line[0]))
    f.close()

show_csv()