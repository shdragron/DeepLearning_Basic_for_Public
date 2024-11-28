# 4_5_csv.py

import csv
import re
import requests

# us-500.csv 파일을 출력하세요


# def show_csv():
#     f = open("data/us-500.csv", 'r', encoding="utf-8") #'w' 초기화 후 쓰기모드
#     for line in csv.reader(f):
#         print(line)
#     print(type(line[0]))
#     f.close()


# show_csv()

def write_csv():
    f1 = open("data/weather_1.csv", 'r', encoding="utf-8") #'w' 초기화 후 쓰기모드
    f2 = open("data/weather_1_copy.csv", 'w', encoding="utf-8", newline='') #'w' 초기화 후 쓰기모드
    rows = []
    for line in csv.reader(f1):
        rows.append(line)
    list = [[0,1,3],[3,2,1]]
    csv.writer(f2, delimiter=',', quoting=csv.QUOTE_ALL).writerows(rows)

    f1.close()
    f2.close()

write_csv()
