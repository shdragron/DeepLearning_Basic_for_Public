# 4_4_weather.py
import re
import requests
url = "http://www.kma.go.kr//weather/forecast/mid-term-rss3.jsp?=stnId=108"

response = requests.get(url)
# empty = re.findall(r'<province>(.+)</province>', response.text)  # ()의미는 여기만 리스트에 넣겠다.
# print(set(empty))
# print(empty[0])
# print(type(set(empty)))

# 데이터를 가져올건데 우리가 원하는 데이터느 로케이션 안 데이터는 많아서 데이터 싱크 작업

# 퀴즈 location을 읽어 오세요.
# location = re.findall(r'location wl_ver="3">(.+)</location>', response.text, re.DOTALL)  # ()의미는 여기만 리스트에 넣겠다.
# print(len(location))

# 같은 줄에 있는 것이 옵션
# 욕심이 많아서 가장 마지막 조합(가장 큰조합)을 찾는다 따라서 1개 +
# .+?: 가장 짧은 패턴 검색(non-greedy)
# .+: 가장 긴 패턴 검색(욕심쟁이,greedy)
# DOTALL: 찾고자 하는 것이 여러줄에 걸쳐있을 때

location = re.findall(r'location wl_ver="3">(.+?)</location>', response.text, re.DOTALL)  # ()의미는 여기만 리스트에 넣겠다.
print(len(location))
print(location[0])

# 퀴즈:  Province를 출력하세요.
province = []
for i in range(len(location)):
    tmp = location[i]
    province1 = re.findall(r'<province>(.+?)</province>', tmp, re.DOTALL)
    print(*province)
    province.append(province1)
# 퀴즈: City를 출력하세요.
city = []
for i in range(len(location)):
    tmp = location[i]
    city1 = re.findall(r'<city>(.+?)</city>', tmp, re.DOTALL)
    city.append(city1)
    print(*city)


# 퀴즈: Data를 출력하세요.
# mode, tmEf, wf, tmn, tmx 출력
# print(len(location))

data = []
mode = []
tmEF = []
tmx = []
wf = []

for i in range(len(location)):
    tmp = location[i]
    data1 = re.findall(r'<data>(.+?)</data>', tmp, re.DOTALL)
    data.append(data1)
    for j in range(12):
        mode1 = re.findall(r'<mode>(.+?)</mode>', data1[j], re.DOTALL)
        tmEF1 = re.findall(r'<tmEF>(.+?)</tmEF>', data1[j], re.DOTALL)
        tmx1 = re.findall(r'<tmx>(.+?)</tmx>', data1[j], re.DOTALL)
        wf1 = re.findall(r'<wf>(.+?)</wf>', data1[j], re.DOTALL)
    mode.append(mode1)
    tmEF.append(tmEF1)
    tmx.append(tmx1)
    wf.append(wf1)

print(len(mode),len(province),len(city), len(tmEF), len(tmx))
print(1)
# print(len(city))
for i in range(41):
    print(*province[i], ',', *city[i], ',',*mode[i],',',*wf[i],',', *tmEF[i], ',', tmx[i] )
# print(len(province))

    # tmp = location[i]
    # city = re.findall(r'<city>(.+?)</city>', tmp, re.DOTALL)
    # print(*city)


def write():
    f = open("data/weather.csv", 'w', encoding="utf-8") #'w' 초기화 후 쓰기모드
    # for i in range(41):
    #     str1 = str(*province[i]) + ',' + str(*city[i]) + ',' + str(*mode[i])+ ',' +str(*wf[i])+ ',' +str(*tmEF[i])+','+str(*tmx[i])+'\n'

    for i in range(41):
        print(*province[i], *city[i], *mode[i], *wf[i], *tmEF[i], *tmx[i],sep = ',',file=f)
        # f.write(str1)
    f.close()
write()

# location = re.findall(r'location wl_ver="3">(.+?)</location>', response.text, re.DOTALL)

# 제주 서귀포 A02 2024-07-22 00:00 구름많음 27 30 20