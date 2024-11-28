// 3_2_for.c
#include <stdio.h>
void f_1();

int main(){
    f_1();
}
//아스키코드 설명
void f_1(){
    int i;
    // char c ='A';
    // printf("%c\n %d\n",c,'A');
    // printf("%c\n %d\n",65,65);
    // printf("%c\n %d\n",c,'a');
    // printf("%c\n %d\n",97,97); 

    //퀴즈: 대문자 전체를 출력하세요.

    // for ( i = 0; i < 26; i++)
    // {
    //     printf("%c\n",a+i);
    // }
    //퀴즈: 소문자 전체를 거꾸로 출력하세요.

    // for (i = 'a'; i >= 'z'; i--)
    // {
    //     int b = (36+i);
    //     printf("%c\n",b);
    // 
    
    //퀴즈
    // 소문자와 대문지 전체를 번갈아 출력하세요(2가지)

    // for (i = 'a'; i <= 'z'; i++)
    // {
    //     printf("\n%c",i);
    //     printf("\n%c",i-32);
    // }

    for (i = 'A'; i <= 'Z'; i++)
    {
        printf("\n%c",i+32);
        printf("\n%c",i);
    }


}


//  
