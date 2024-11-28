// switch_5_1.c

#include <stdio.h>
#include <stdlib.h>

void f_1();
// int* backup();
void f_1();

int main(){
    f_1();
    f_2();
    // int b ;
    // printf("%p", (void*)f_1);
    // int* t = backup();
    // printf("%d,%d", t[0],t[4]);
    // return 0;
}

void f_1(){
    int n= 99;
    printf("양수 입력: ");
    scanf("%d", &n);
    // 입력한 양수의 나머지를 알려주세요.
    printf("%d\n",n%3);
    // if(n%3 == 0){ -> 스위치 문으로 변경 가능

    // }
    // else if(n%3 == 1){

    // }
    // else(n%3 == 2){

    // }
    switch (n%3)
    {
    case 0:
        printf("0");
        break;
        
    case 1:
        printf("1");
        break;
    default:
        printf("2");
       break;
    }
}

// int*  backup(){ -> 이렇게 사용하면 안됨.
//     int a[5] = {3, 5, 7, 9, 1};
//     return a;
// }

// int my_scanf(int n){ -> 가변인자 불가능. 
//     return n;
// }
// 퀴즈: 점수를 학점으로 변환하는 함수오 만드세요.
// void f_2(){
//           break;
    
//     default:
//         break;
//     }  switch (a)
//     {
//     case 90 < :

// }