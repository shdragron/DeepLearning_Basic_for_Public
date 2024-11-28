# 6_2_sqlite.py

import sqlite3
import csv


# 퀴즈: weather.csv 파일을 반환하는 함수를 만드세요.

# def return_csv():
#     f1 = open("data/weather_1.csv", 'r', encoding="utf-8")  # 'w' 초기화 후 쓰기모드
#     rows = []
#     for line in f1:
#         rows.append(line.split(','))
#     f1.close()
#     return rows


# print(return_csv())

def return_csv_advanced():
    with open("data/weather_1.csv", 'r', encoding="utf-8") as f:
        return [line.split(',') for line in f]

weather = return_csv_advanced()
 # 원래 db는 링크를 달고 로그인 등을 할 수 있다.
db_file = "data/weather.sqlite" # 확장자명은 sqilite 마음대로 -> 바이너리 파일이다.
# 1)create
def create_table(db_file): # db를 변화시킨다.
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    query = "CREATE TABLE kma (prov TEXT, city TEXT, mode TEXT, tmEF TEXT, wf TEXT, tmn TEXT, tmx TEXT, rnSt TEXT)"  # 구조화된 질문 언어 -> 커맨드이다.
    cur.execute(query)

    conn.commit()   # 변경 사항이 있으면 커밋을 해야된다.
    conn.close()
# 2)read
def insert_row(row): # db 변화시킨다.

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    query = 'INSERT INTO kma VALUES("{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}")'
    cur.execute(query.format(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7]))

    conn.commit()   # 변경 사항이 있으면 커밋을 해야된다.
    conn.close()

def insert_rows(rows): # 더 빠르게 하는법 전체를 넣기.

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    query = 'INSERT INTO kma VALUES("{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}")'

    for row in rows:
        # cur.execute(query.format(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7]))
        cur.execute(query.format(*row))

    conn.commit()   # 변경 사항이 있으면 커밋을 해야된다.
    conn.close()
# 2)read
def fetch_all(db_file):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    query = "SELECT * FROM kma"

    # for row in cur.execute(query):
        # print(row) # db데이터는 튜플로 나와서 변경이 불가하다.

    rows = [row for row in cur.execute(query)]

    conn.commit()   # 변경 사항이 있으면 커밋을 해야된다.
    conn.close()

    return list(rows)

# 퀴즈: city가 부산인 모든 레코드를 보여주세요.

def search_all(db_file, city):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    query = 'SELECT * FROM kma WHERE city = "{}"'.format(city)

    # for row in cur.execute(query):
        # print(row) # db데이터는 튜플로 나와서 변경이 불가하다.

    rows = [row for row in cur.execute(query)]

    conn.commit()   # 변경 사항이 있으면 커밋을 해야된다.
    conn.close()

    return rows

weather = return_csv_advanced()

# for row in weather:
    # insert_row(row)

# insert_rows(weather)

rows = fetch_all(db_file)

rows2 = search_all(db_file,'부산')
print(rows2,sep = '\n')


# create_table(db_file)