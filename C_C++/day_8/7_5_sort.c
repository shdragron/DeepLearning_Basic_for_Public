// 7_5_sort.c

#include <stdio.h>
#include <string.h>

void int_change(int* a, int* b);
void show_array(int a[], int size1);
void bubble_sort(int*  p1, int size);
void bubble_up(int*  p1, int size);
int max_pos(int* p1, int size);
void insert_sort(int*  p1, int size);
void char_insert_sort(const char*  names, int size);
void char_change(const char** a,const  char** b);
void show_names(const char* names[], int size1);
int max_pos_names(const char* names[], int size);


int main(){
    int a = 7;
    int b = 3;

    int_change(&a,&b);
    // printf("%d %d",a,b);


    int ages[10] = {15, 90, 83, 34, 72, 10, 51, 22, 66, 45};
    // int_change(&ages[0],&ages[1]);
    // int_change(ages,ages+1);

    // show_array(ages,10);
    // show_array(ages, 10);
    // bubble_sort(ages, 10);
    // show_array(ages, 10);
    // max_pos(ages, 10);
    // printf("%d", max_pos(ages, 10));
    insert_sort(ages,10);
    show_array(ages, 10);

    const char* name [] = {"kim","lee","han","nam","koo"};
    char_change(name[0], name[1]);
    show_names(name,5);




    return 0;
}

// 퀴즈: 중수 2개를 교환하는 함수를 만드세요.

void int_change(int* a, int* b){
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void show_array(int a[], int size1){
    for(int i = 0; i < size1; i++){
        printf("%d ",a[i]);
    }
    printf("\n");
}


// 거품 정렬
// 10, 15, 90, 83, 34

void bubble_up(int*  p1, int size){
    for (int i = 0; i < size - 1; i++)
    {
        if( p1[i] > p1[i+1]){
            int_change(&p1[i],&p1[i+1]);
        }
    }
}

void bubble_sort(int*  p1, int size){
    for (int i = 0; i < size - 1; i++)
    {
        bubble_up(p1,size);
    }
    
}

int max_pos(int* p1, int size){
    int max = 0;
    for (int i = 0; i < size; i++){
        if( p1[max] < p1[i]){
            max = i;
        }
    }
    return max;
}

//  왜 안되는지 여쭤보기

void insert_sort(int*  p1, int size){

    int tmp = 0;

    for (int i = size-1; i > -1; i--)
    {
        tmp = p1[i];

        p1[i] = p1[max_pos(p1, i+1)];

        p1[max_pos(p1, i+1)] = tmp;

    }
    
}

// 캐릭터는 모르갰다. 문제도 이해를 못하겠어.

void char_insert_sort(const char*  names, int size){

    int tmp = 0;

    for (int i = size-1; i > -1; i--)
    {
        tmp = max_pos_names(names,i);
        char_change(names+tmp, names+i-1);

    }
    
}


void show_names(const char* names[], int size1){
    for(int i = 0; i < size1; i++){
        printf("%s ",names[i]);
    }
    printf("\n");
}


int max_pos_names(const char* names[], int size){
    int max = 0;
    for (int i = 0; i < size; i++){
        if(strcmp( names[max] , names[i]) < 0){
            max = i;
        }
    }
    return max;
}

void char_change(const char** a,const  char** b){
    const char* tmp = *a;
    *a = *b;
    *b = tmp;
}