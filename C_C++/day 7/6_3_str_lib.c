// 6_3_stack.c

// 자료구조(Data Structure)
// stack, que, linked list
// stack은 식장 휴지랑 같다.

#include <stdio.h>
#include <string.h>



int main(){

    const char* s1 = "hello";
    const char* s2 = "hi";

    // printf("%d\n",strcmp(s1,s2)); // s1 - s2 = -4 즉 s2가 더 크다.. 사전순서로 크기가 나뉨.

    //  퀴즈: strchr 함수를 사용해서 특정 문자가 몇개인지 알려주세요.

    const char* p = "strawberry is very sweet";

    // const char* p1 = strchr(p,'r');
    // const char* p2 = strchr(&p1[1],'r');
    // const char* p3 = strchr(&p2[1],'r');
    // const char* p4 = strchr(&p3[1],'r');
    
    // int j = 0;
    // char i1;
    // scanf("%c",&i1);

    // while(p != NULL){
    //     p = strchr(p, i1);

    //     if(p != NULL){
    //         printf("%s",p);
    //         p = &p[1];
    //         j++;

    //     }
    // }

    int j = 0;
    char i1;
    scanf("%c",&i1);

    while(p = strchr(p,i1))
        printf("%s\n", p++);
        j++;

    printf("\n개수 : %d개",j);

    // printf("%s\n",p1);
    // printf("%s\n",p2);
    // printf("%s\n",p3);
    // printf("%s\n",p4);

    return 0;
}
