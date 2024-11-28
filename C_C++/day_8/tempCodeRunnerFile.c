// 7_4_string_array.c

#include<stdio.h>
#include<string.h>

void show_array(char word[][256]);
int biggest(char word[][256]);


int main(){

    char words[3][256] = {"red", "green","blue"};

    show_array(words);
    printf("\n%d is biggest\n",biggest(words));    

    int idx = -1;
    idx = strcmp(words[0],words[1]) > 0 ? 0 : 1;
    idx = strcmp(words[idx],words[2]) > 0 ? idx : 2;
    puts(words[idx]);
    return 0;

}

// 퀴즈: (문자열 배열 -> 2D) 를 출력하는 함수를 만드세요.

void show_array(char word[][256]){
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