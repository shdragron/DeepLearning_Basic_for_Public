// hello.c
#include <stdio.h>
void f_1();
void f_2();
void f_3();

int main() {
   
    f_1();
    f_2();
    f_3();
}

// 퀴즈 "rainy day" 출력 3번(3가지 코드)

void f_1() {
    //1
    printf("abc\n");
    printf("abc\n");
    printf("abc\n");
    //2
    printf("abc abc abc \n");
    //3
    printf("abc" "abc" "abc\n");
    //4
    printf("a");
    printf("b");
    printf("c\n");
    printf("a");
    printf("b");
    printf("c\n");
    printf("a");
    printf("b");
    printf("c\n");
    //5
    printf("%d%d%d\n", 1, 2, 3);
}   

void f_2() {
    printf("%c %d %f %s\n", 'a', 12, 3.14,"ace");
    int a =144;
    printf("%d",a);
    printf("printf : %p\n", printf);
    printf("%d\n", printf("43"));
}

void f_3(){
    int a =3, b =7;
    printf("%d %d\n", a,b);
    int c = a;
    a = b;
    b = c;
    printf("%d %d", a,b);
}

