# 1_2_imdb.py
import requests
import re
# 퀴즈
# imdb top250 사이트의 데이터를 가져오세요.
# def read():
    f = open("data/top100.txt", 'r', encoding="utf-8") #'w' 초기화 후 쓰기모드
    str1 = f.read()
    return str1
    f.close()
#
    songs = re.findall(r'<div class="o-chart-results-list-row-container">(.+?)<div class="lrv-u-flex-grow-1 lrv-u-height-100p lrv-u-padding-l-2 lrv-u-padding-r-050 lrv-u-padding-lr-00@mobile-max">',read(), re.DOTALL)
print(len(songs))

for song in songs:
    # print(loc)
    title = re.findall(r'<h3 id="title-of-a-story" class="c-title  a-no-trucate a-font-primary-bold-s u-letter-spacing-0021.+>.+(.+?).+</h3>', song, re.DOTALL)
    singer = re.findall(r'<span class="c-label  a-no-trucate a-font-primary-s lrv-u-font-size-14@mobile-max>.+(.+?).+</span>', song, re.DOTALL)
    print(title[0], singer[0])

    # data = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)


# 퀴즈
# hot100.txt 파일을 읽고 순위, 제목, 가수를 출력하세요
# f = open('data/top100.txt', 'r', encoding='utf-8')
#
# text = f.read()
# # print(text)
# # print(len(text))
#
# #
# pattern = r'<h3 id="title-of-a-story" class="c-title  a-no-trucate a-font-primary-bold-s u-letter-spacing-0021 '\
#           r'.+?>(.+?)</h3>'
# h3 = re.findall(pattern, text, re.DOTALL)
#
# pattern = r'<span class="c-label  a-no-trucate a-font-primary-s .+?>(.+?)</span>'
# span = re.findall(pattern, text, re.DOTALL)
#
#
# for title, singer in zip(h3, span):
#     print(title.strip(), ':', singer.strip())
#
# f.close()