#include <stdio.h>

int f_2(int a); // -> 주의! 매개변수 넣은 상태에서 함수 선언!

int main(){
    f_2(3); // 함수 호출
    printf("%d",f_1(3));
}
int f_2(int a){
    return a; //반환 값
}
