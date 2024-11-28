#include <stdio.h>
void f_2();


void f_1() {
    // //관계 : > >= < <= == !=
    // printf("%d\n", 7>3); //1(TRUE)
    // printf("%d\n", 7<3); //0(FALSE)

    // //퀴즈
    // //나이를 입역 받아서 10대인지 알려주세요.(without 비트 연산)
    // int a;
    // scanf("%d",&a);
    // printf("%d\n", ((int)(10<= a) + (int)(a < 20))/2);
    // //printf("%d\n", ((10<= a) * (a < 20)));
}

#define TRUE 1
#define FALSE 0

// void f_3(){
//     printf("%d %d\n", TRUE, FALSE); // 1,0
//     printf("%d\n",TRUE && TRUE); // 1
//     printf("%d\n",TRUE && FALSE);//0
//     printf("%d\n",FALSE && TRUE);//0
//     printf("%d\n",FALSE && FALSE);//0
// }

void f_4(){
    int a = 7;
    if(a % 2){ // a % 2 == 1 과 동일.
        printf("odd\n");
    }
    else{
        printf("Even\n");
    }
}

//퀴즈
//어떤 정수가 양수인지 음수인지 0인지 알려주세요.

void f_5(){
    int a;
    scanf("%d", &a);

    if(a > 0){
        printf("양수\n");
    }
    else if(a == 0){
        printf("0\n");
    }
    else{
        printf("음수\n");
    }
}

void f_2(){
    int a;
    char str[10] = "";
    int count = 10;
    int i;
    scanf("%d", &a);
    //퀴즈
    //점수를 학점으로 바꿔주세요.
    if(a <= 100 && a>= 90){
        char temp[10] = "A grade!";
        for ( i = 0; i < count; i++)
        {
            str[i] = temp[i];
        } 
    }
    else if(a <= 89 && a>= 70){
        char temp[10] = "B입니다";
        for ( i = 0; i < count; i++)
        {
            str[i] = temp[i];
        }
    }
    printf("%s\n",str);

}

int f_7(int a, int b){
 
    //퀴즈
    //점수를 학점으로 바꿔주세요.
    if(a <= 100 && a>= 90){
        b = 1;
    }
    else if(a <= 89 && a>= 80){
        b = 2;
    }
    else if(a <= 79 && a>= 70){
        b =3;
    }
    else if(a <= 69 && a>= 50){
        b =4;
    }
    else if(a <= 49 && a>= 0){
        b = 5;

    }
    return b;

}

int main(){
    int a;
    int b = 10;
    scanf("%d", &a);
    printf("%d",f_7(a,b));

}