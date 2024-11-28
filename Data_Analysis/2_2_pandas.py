# 2_2_pandas.py
import pandas as pd

s = pd.Series([1, 2, 3, 4, 5])
print(s)

print(s.index)
print(s.values)
print(type(s.values))
print(s[4])


s.index = list('abcde')
print(s['a'])
print(s['e'])

# 퀴즈 앞 value 2개만 출력하세요
print()
print(s[:2].values)
print(s[:'b'].values) # 레이블에는 대소가 없어서 포함된다.
print(s.values[:2])

print()
info = {
    'year': [2021, 2022, 2023, 2021, 2022, 2023],
    'city': ['pusan', 'pusan', 'pusan', 'jeju', 'jeju', 'jeju'],
    'population': [500, 600, 700, 300, 400, 350],
}

# 퀴즈: 2021년도의 지역 이름과 인구 수를 출력하세요.

print(info.get('year'))
for i in zip(info.get('year'),info.get('population'),info.get('city')):
    if i[0] == 2021:
        print(i[1], i[2])

print()
for i in range(len(info['year'])):
    if info['year'][i] == 2022:
        print(info['city'][i])
        print(info['population'][i])

# info['city']
# info.get('city') 차이점은 get은 없는 키값도 생성해준다.
print()
df = pd.DataFrame(info)
print(df)
print(df.index)
print(df.columns)
print(df.values)
print(df.values.shape)
print(df['year'])
print(df.year) # 위 와같은데 컬럼 삭제시 에러 발생 유도

print()
df.index = list('abcdef')

print(df.iloc[0])
print(type(df.iloc[1])) # Series로 열이다.

print(df.loc['c'])

# 퀴즈: pusan 데이터만 출력하세요.

# for i in (df.values):
#     if(i[1] == 'pusan'):
#         print("city",i['city'],"year:", i['year'], "population:",i['population'])

# 퀴즈: 2022년 jeju의 인구를 구하세요.
print([i[2] for i in df.values if i[0] == 2022 and i[1] == 'jeju'])
