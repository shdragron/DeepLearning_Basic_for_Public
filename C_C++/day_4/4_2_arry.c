// 4_2_arry.c
#include <stdio.h>
#include <stdlib.h>


void randon(int list[],int size);

void make_randoms(int a[], int size){ //배열 리턴하는 방법
    srand(3);
    int i = 0;
    for ( i = 0; i < size; i++)
    {
        a[i] = rand()%100;
    }
}

int biggest(int a[],int size){
    int i = 0;
    int max = a[0]; //0; // 이러면 틀린다. 안에 있는 임의 값을 적용하는게 맞다.
    for ( i = 0; i < size; i++)
        if(a[i] > max){
            max = a[i];
        }
    return max;
    
}

int main(){
    int i = 0;
    int a[10];
    int b[10];
    make_randoms(a,10);
    randon(b,sizeof(b)/sizeof(b[0])); //의미: 총 바이트/ (자료형 바이트) = 개수

for ( i = 0; i < sizeof(a)/sizeof(a[0]); i++)
{
    printf("%d ",a[i]);
}
printf("\n");
for ( i = 0; i < sizeof(b)/sizeof(b[0]); i++)
{
    printf("%d ",b[i]);
}
printf("\n제일 큰 값: %d",biggest(a,sizeof(a)/sizeof(a[0])));
printf("\n제일 큰 값: %d",biggest(b,sizeof(b)/sizeof(b[0])));
    
}

void randon(int list[], int size){
    int i = 0;
    for (i = 0; i < size; i++)
    {
        list[i] = rand()%100;
    }
    
}

// 내가 잘못 이해 했던 함수
// int random(){
//     int a[10] = {0,0,0,0,0,0,0,0,0,0};
//     int i = 0;
//     for ( i = 0; i < 10; i++)
//     {
//         a[i] = rand();
//     }
//     return a;
// }







// int main(){
//     int list1[4] = {1,2,3,4};
//     int size = sizeof(list1)/4;
//     show_arry(list1,size);
//     return 0;

// }

// void show_arry(int list[],int size){
//     // int a[3] = {0,2,4};
//     // printf("%d,%d,%d\n",a[0],a[1],a[2]);
//     // 퀴즈 배열을 출력력하는 함수를 만드세요.
//     int a = 0;
//     int i = 0;
//     int max = 0;
//     printf("배열: %p\n",list);
//     printf("함수: %p\n",printf);
//     printf("Sizeof 함수: %p\n",sizeof(list));
//     for(i = 0; i < size; i++){
//         printf("%d",list[i]);
//     }
// }