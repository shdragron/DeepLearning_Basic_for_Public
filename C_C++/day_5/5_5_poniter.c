// 5_5_pointer.c

#include <stdio.h>
#include <string.h>

void swap(int* a, int* b);

//퀴즈: 변수 2대의 닶을 교환한 함수를 만드세요. (swap)

int main(){
    int a = 7, b = 3;
    printf("%p, %p\n", &a,&b);
    swap(&a, &b);
    printf("%d %d\n",a,b);
    printf("%p, %p\n", &a,&b);
    return 0;
    int *c = &a;
    printf("%d",c);
    c = a;


}

void swap(int *a, int * b){
    printf("%p, %p\n", a,b);
    int tmp;
    tmp = *a; //*은 해당주소 값의 데이터
    *a = *b;
    *b = tmp;
    printf("%p, %p\n", a,b);
}

void swap2(int *a, int *b){
    printf("%p, %p\n", a,b);
    int tmp;
    tmp = a[0];
    a[0] = b[0];
    b[0] = tmp;
    printf("%p, %p\n", a,b);
}