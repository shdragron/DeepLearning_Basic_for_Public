#include <stdio.h>
#include <string.h>

int main(){

    char s1[256] = "abc"; //배열을 할당
    char s2[256] = {'d','e','f'}; //배열을 할당
    char *s3 = "ghi"; //주소를 저장할 8바이트만을 할당

    //s1 = s2; : 주소가 상수 (에러)
    s3 = s2; // 주소가 변수


    // s1[0] = 'A'; : 값은 변수
    // s2[0] = 'G'; : 값은 상수(에러)

    printf("%p %s\n", s3,s3);

    // printf("%s %s %s\n", s1, s2, s3); //시용하기 편한 배열
    //문자열: 커피 무지개  파인애플 -> 크기가 256인 이유는? -> 넉넉하게 잡은거임. -> 충분히 크다. (대부분의 문자열을 담을 수 있다.)
    // int i = 0;
    //     for ( i = 0; i < strlen(s1); i++){
    //         printf(" %c %c %c\n", s1[i],s2[i],s3[i]);
    //     }
    // return 0;
    // for ( i = 0; i < 256; i++)
    // {
    //     if(s1[i]==0 && s2[i]==0 && s3[i]==0){
    //         break;
    //     }
    //     else{
    //         printf(" %c %c %c", s1[i],s2[i],s3[i]);
    //     }
    // }
    // 1
    // int j = 0;
    // while(s1[j]!= '\0'){ // a b c \0
    //     putchar(s1[j++]);
    // }

    // int j = 0;
    // while(s1[j]){ // a b c \0
    //     putchar(s1[j++]);
    // }
    

    // char *p = s1;
    // while(*p){
    //     putchar(*p++);
    // }
    // j ++ + ++j -> 이 결과는 컴파일러 마다 달라서 위험. 사용하지 않는다.
}