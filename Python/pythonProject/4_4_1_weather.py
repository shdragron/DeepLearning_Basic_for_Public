# 4_4_weather.py
import re
import requests

f = open("data/weather_1.csv", "w", encoding="utf-8")

# 퀴즈
# 기상청 아래 주소로부터 province만 출력하세요
url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108"
response = requests.get(url)
# print(response)
# print(response.text)

# province = re.findall(r'<province>(.+)</province>', response.text)
# print(*province, sep='\n')

# 퀴즈
# location을 읽어오세요
# .+ : 가장 긴 패턴 검색(욕심쟁이, greedy)
# .+? : 가장 짧은 패턴 검색(non-greedy)
# DOTALL: 찾고자 하는 것이 여러 줄에 걸쳐있을 때 사용
location = re.findall(r'<location wl_ver="3">.+?</location>', response.text, re.DOTALL)
# print(len(location))
# print(location[0])

# 퀴즈
# province만 출력하세요
# city만 출력하세요
# data만 출력하세요
# mode, tmEf, wf, tmn, tmx, rnSt만 출력하세요
for loc in location:
    # print(loc)
    prov = re.findall(r'<province>(.+?)</province>', loc)
    city = re.findall(r'<city>(.+)</city>', loc)
    print(prov[0], city[0])

    data = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
    # print(len(data))

    for datum in data:
        mode = re.findall(r'<mode>(.+)</mode>', datum)
        tmEf = re.findall(r'<tmEf>(.+)</tmEf>', datum)
        wf = re.findall(r'<wf>(.+)</wf>', datum)
        tmn = re.findall(r'<tmn>(.+)</tmn>', datum)
        tmx = re.findall(r'<tmx>(.+)</tmx>', datum)
        rnSt = re.findall(r'<rnSt>(.+)</rnSt>', datum)

        # print(prov[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0])
        print(prov[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0],
              sep=',', file=f)

        # f.write("{},{},{},{},{},{},{},{}\n".format(prov[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0]))

        # item = prov[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0]
        # f.write("{},{},{},{},{},{},{},{}\n".format(*item))

f.close()

# 제주도 서귀포
# A02 2024-07-22 00:00 구름많음 27 30 20
# A02 2024-07-22 12:00 구름많음 27 30 30
# =>
# 제주도 서귀포 A02 2024-07-22 00:00 구름많음 27 30 20
# 제주도 서귀포 A02 2024-07-22 12:00 구름많음 27 30 30

# 퀴즈
# 파싱한 내용을 파일로 저장하세요 (weather.csv)





