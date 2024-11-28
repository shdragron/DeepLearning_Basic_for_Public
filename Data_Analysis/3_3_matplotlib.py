# 3_3_matplotlib.py

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

# matplot 한글 적용
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


# 퀴즈: 2016_GDP.txt 파일을 반환하는 함수를 만드세요.

def read_gdp():
    # data = pd.read_csv('data/2016_GDP.txt',
    #                 delimiter=':', engine='python')
    # print(data)
    f = open(f"data/2016_GDP.txt",'r',encoding='utf-8')
    f.readline() #skip header 제일 앞에 있는 이름들을 삭제
    names, dollars = [], []
    for line in f:
        # print(line.strip().split(':'))
        _, name, dollar = line.strip().split(':')
        dollar = dollar.replace(',','')
        dollar = int(dollar)
        names.append(name)
        dollars.append(dollar)
    f.close()

    return names, dollars


print(read_gdp())

# 퀴즈: top10에 대해서 막대그래프로 그려주세요.
names, dollars = read_gdp()
print(type(dollars[0]))
n10 = names[:10]
d10 = dollars[:10]
print(d10)

# print(names)
x = np.arange(len(d10))
print(x)
plt.bar(x, d10, color=['#FFA07A','#CD5C5C'], width=0.25)
plt.bar(x + 0.25, d10, color= colors.TABLEAU_COLORS, width=0.25, alpha=0.5) # alpha는 투명도
plt.title('GDP') # 표 제목
plt.xticks(x+0.125, n10, rotation=45) # legend 수정
plt.subplots_adjust(bottom=0.2, top=0.8) # 전체적인 표 위치 지정
plt.show()
