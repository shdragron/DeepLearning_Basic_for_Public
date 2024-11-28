//3_1_function.c
#include <stdio.h>

//전역변수: 코드 전체에서의 변수
//지역변수: 함수내 변수

int f_1(int c, int d);

int main(){
    int a;
    int b;
    scanf("%d\n%d", &a, &b);
    f_1(a,b);
    printf("%d", f_1(a,b));
}
//퀴즈
// 큰값을 나오는 함수

int f_1(int c, int d){
    // if(c >= d){
    //     return c;
    // }
    // else {
    //     return d;
    // }
    //고수들이 쓰는법
    if(c >= d){
        d = c;
    }
        return d;
}

//퀴즈
// 정수의 값을 2배로 만들어주는 함수를 만드세요

// int f_1(int c){
//     return c*2;
// }
