// hello.c
#include <stdio.h>

void f_1();

int main() {
    f_1();
}
//연산: 산술, 관계, 논리, 비트

void f_1(){
    int a = 10;
    int b = 3;

    printf("%d\n",a+b);
    printf("%d\n",a-b);
    printf("%d\n",a/b);
    printf("%d\n",a%b);
    printf("%d\n",a*b);

    //퀴즈 스왑변수를 사용하지 안고 구현하기
    a = a + b;
    b = a - b;
    a = a - b;

    printf("%d %d", a,b);
}