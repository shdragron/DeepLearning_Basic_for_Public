//3_3_while.c
#include <stdio.h>

int f_1(int a, int b);

int main(){

    printf("%d",f_1(100,100));
}

int f_1(int a, int b){
    int tmp = 0;

    // int tmp = 0;
    // int sum = 0;
    // //퀴즈
    // //1부터 10까지의 합계를 구하시오
    // while(tmp < 11){
    //     sum += tmp;
    //     tmp++;
    // }
    // printf("%d\n",sum);

    //퀴즈
    //정수 범위에 대해 합계를 구하는 함수를 만드세요.

    if(a < b){
        while (a < b){
            a++;
            tmp += a;   
        }
    }
    else if(a > b)
        while (a > b){
            b++;
            tmp += b;
        }
    }
    else {
        tmp = a;  
    }
return tmp;
}