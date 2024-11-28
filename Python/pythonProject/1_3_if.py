# 1_3_if.py

#퀴즈
#어떤 양수가 홀수인지 짝수 인지 알려주세요.


# if a % 2 == 1:
#     print("양의 홀수")
#
# else:
#     print("양의 짝수")

# 퀴즈
# 어떤 양수가 3의 배수인지 5의 배수인지 3과 5의 배수인지 알려주세요
a = int(input("양수: "))

if a % 3 == 0:
    if a % 5 ==0:
        print("3과 5의 배수")
    else:
        print("3의 배수")
elif a % 5 == 0:
    print("5의 배수:")
else:
    print("해당되지 않음")


