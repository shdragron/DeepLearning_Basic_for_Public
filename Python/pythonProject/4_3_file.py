# 4_3_file.py
import re
#절대 경로
#거의 쓰지 않는다.
#상대위치
file_path2 = "data/poem.txt"

# def read_1(file_path): #읽기
#     f = open(file_path, "r", encoding="utf-8")
#     poem = f.read()
#     print(poem)
#     rows = poem.split('\n')
#     print(rows)
#     print(*rows, sep='\n')
#     print('\n'.join(rows))
#     return poem
#     f.close()


def read_2(file_path): #읽기
    f = open(file_path, "r", encoding="utf-8")
    # poem = f.read(10)
    # print(poem)

    lines = f.readlines()
    print(*lines)
    print()
    for line in lines:
        # print(line)
        print(line.strip("\n"))
    return lines

    f.close()

def read_3(file_path): #읽기
    f = open(file_path, "r", encoding="utf-8")
    # poem = f.read(10)
    # print(poem)

    while(True):
        line = f.readline()
        if line:
            break
        print(line)

    f.close()

def write():
    f = open("data/sample.txt", "w", encoding="utf-8") #'w' 초기화 후 쓰기모드
    f.write("아프니까 청춘이다. ")
    f.write("\n")
    f.write("나이든 사람도 아프다.")

    f.close()

response_text = read_3(file_path2)
print("-"*100)
# print(response_text)
# print(type(response_text))
#
# s = "{} {} {} {}".format(12, 3.14, "rain", True)
# print(s)

# t= """
#
# \t\t\t hello world \t\t\t
#
# """
# print("[{}]".format(t))
# print("[{}]".format(t.strip()))

# print(type(s))

# 퀴즈: 파일에 포함된 개행문자의 갯수를 알려주세요.

# # 1)
# a = [i for i in response_text if i == '\n']
# print(a)
# print("개행문자 갯수",len(a))
#
# # 2)
# empty = re.findall(r'(\n)', response_text) # ()의미는 여기만 리스트에 넣겠다.
# print(empty)
# print("개행문자 갯수: ",len(empty))

read_3(file_path2)
write()




