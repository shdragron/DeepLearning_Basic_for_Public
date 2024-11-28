// 7_4_string_array.c

#include<stdio.h>
#include<string.h>

void show_array(char word[][256]);
int biggest(char word[][256]);
void p_r(char* colors[],int size);


int main(){

    char words[3][256] = {"red", "green","blue"};

    show_array(words);
    printf("\n%d is biggest\n",biggest(words));    

    int idx = -1;
    idx = strcmp(words[0],words[1]) > 0 ? 0 : 1;
    idx = strcmp(words[idx],words[2]) > 0 ? idx : 2; // 참이면 idx사용 거짓이면 2
    // puts(words[idx]);

    // char s1[256] = "string"; // 문자열 리시트 값 할당
    // char* s2 = "words"; // 주소 값 할당

    //퀴즈: 아래와 같은 형태의 문자열 배열을 출력하는 함수를 만드세요.
    char* colors[3] = {"red", "green","blue"};
    p_r(colors,3);

    return 0;

}

// void p_r(char* colors[],int size){
void p_r(char** colors,int size){

    for (int i = 0; i < size; i++)
    {
        puts(colors[i]);
    }
    
}

// 퀴즈: (문자열 배열 -> 2D) 를 출력하는 함수를 만드세요.

// void show_array(char word[][256]){
void show_array(char(* word)[256]){

    for (int i = 0; i < 3; i++){
        puts(word[i]);
    }
}


// 퀴즈: 문자열 배열에서 가장 큰 문자열은 누구인지 알려주세요.

int biggest(char word[][256]){
    if(strcmp(word[0],word[1]) >= 0){
        if(strcmp(word[0],word[2]) >= 0){
        return 0;
        }
        else{
            return 2;
        }
    }
    else if(strcmp(word[0],word[1]) < 0){
        if(strcmp(word[1],word[2]) >= 0){
        return 1;
        }
        else{
            return 2;
        }
    }
}