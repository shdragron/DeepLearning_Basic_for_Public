# 1_2_imdb.py
import requests
# 퀴즈
# imdb top250 사이트의 데이터를 가져오세요.

header = {
 'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
}
url = "https://www.imdb.com/chart/top/"
response = requests.get(url, headers=header)
print(response.text)

