#include <stdio.h>

int main() {
    int a = 1;
    int b = 2;
// 퀴즈 "rainy day" 출력 3번(3가지 코드)

//퀴즈 스왑변수를 사용하지 안고 구현하기
    a = a + b;
    b = a - b;
    a = a - b;
    printf("%d %d",a,b);
}