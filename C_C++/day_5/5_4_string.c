// 5_4_string.c
#include <stdio.h>
#include<string.h>

//퀴즈: 문자열의 길이를 계산하는 함수를 만드세요.
//퀴즈: 문자열을 복사하는 함수를 만드세요.
//퀴즈: 문자열에 포함된 문자를 찾는 함수를 만드세요.

int string_char(const char s[],char d);
void string_copy(const char s[],char d[]);
int string_length(const char s[]);


int main(){
    char s[256] = "hello, Gi Yong";
    printf("\n");
    char buffer[256] ="21212121212121212121212";
    char d;
    scanf("%c",&d);
    printf("크기: %d\n",string_length(s));
    printf("크기: %d\n",strlen(s));
    string_copy(s,buffer);
    strcpy(buffer,s);
    printf("복사: %s\n",buffer);
    // printf("여기 있어요: %d\n",string_char(s,d));
    // printf("여기 있어요: %p\n",strchr(s,d));
    return 0;
}

int string_length(const char s[]){

    int j = 0;
    while(s[j]){
        j++;
    }
    return j;

}
void string_copy(const char s[],char d[]){
    while(*d++ = *s++){
        ;
    }
    *d = '\0';
}

void string_copy1(const char s[],char d[]){
    int j = 0;
    while(d[j] = s[j]){
        j++;
    }
    d[j] = '\0';
}

int string_char(const char s[],char d){
    while(1){
        for ( int i = 0; s[i]; i++){
            if(s[i]==d){
                return i+1;
            }
        }
        break;
    }
    return 0;
}