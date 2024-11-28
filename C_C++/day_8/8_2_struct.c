// 8_2_struct.c

#include <stdio.h>

struct INFO {
    int age;
    double weight;
    char gender;
}; // 여기서 가장큰 8바이트로 

void show_info(struct INFO a);
void show_info_advanced(struct INFO* a);
void show_info_array(struct INFO p[],int size);

int main(){
    struct INFO me = {21, 75.0,'M'};

    struct INFO us[] = {
    {20, 50.0, 'F'},
    {50, 60.0, 'M'},
    {30, 80.0, 'M'},
    }; 
    // show_info(me);
    show_info_advanced(&me);
    show_info_array(us,3);
}

// int, char*, double, boolean ==> colection

// 퀴즈: INFO 구조체를 출력하는함수를 만드세요.

void show_info(struct INFO a){
    printf("나이: %d\n", a.age);
    printf("체중: %1.f\n", a.weight);
    printf("성별: %s\n", a.gender == 'M' ? "남자" : "여자");
}

void show_info_advanced(struct INFO* p){ //포인터 버전
    printf("나이: %d\n", (*p).age);
    printf("체중: %1.f\n", p->weight);
    printf("성별: %s\n", p->weight == 'M' ? "남자" : "여자");
}

// 퀴즈 구조체 배열을 출력하는 함수를 만드세요.

void show_info_array(struct INFO p[], int size){ //포인터 버전

    for (int i = 0; i < size; i++)
    {
    // show_info_advanced(&p[i]);
    // show_info_advanced(&*(p+i));
    show_info_advanced(p+i);
    }
}