#include <stdio.h>
#include <stdlib.h>

// void by(int a[])
void copy_a(int a[], int size,int c[]);

int main(){
    int i = 0;
    int a[10] = {1,2,5,6,7,8,8,8,2,10}; // a[10]은 상수
    double b[10] = {1,2,4}; 
    int c[10];
    int* d = c;
    printf("%d\n",c);
    printf("%d",d);

    // printf("%d\n",sizeof(a));// 4바이트 10개: 40
    // printf("%d\n",sizeof(b)); //8바이트 10개: 80
    // // by(b);
    // copy_a(a,sizeof(a)/sizeof(a[0]),c);//a를 c로 복사

    // for(i = 0; i < sizeof(a)/sizeof(a[0]); i++){
    //     printf("%d ",c[i]);
    // }
    // printf("%p",c);
    // printf("\n");
    // for(i = 0; i < sizeof(a)/sizeof(a[0]); i++){
    //     printf("%d ",a[i]);
    // }
    // printf("%p",a);
}

// void by(int a[]){
//     printf("%d",sizeof(a)); //8인 이유는 주소값이여서 8바이트 -> 64비트 -> a[]은 변수

// }

void copy_a(int a[], int size,int c[]){
    int i = 0;
    for(i = 0; i < size; i++){
        c[i] = a[i];
    }
}