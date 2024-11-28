// 7_3_1D_array.c
#include <stdio.h>


void show_def(int a[], int size1);
void copy_1(int grid[], int dst[]);
void copy_2(int src[][5], int dst[]);


int main(){
    int a[3] = {1, 3, 5};
    int b[5] = {2, 4, 6, 8, 0};


    show_def(a,3);
    show_def(b,5);

    //  1 3 5
    //  2 4 6 8 0
    //  p + size
    char s[] = "hello";
    // show_def("s, 5"); -> error ? 정수로 선언되서 4바이트씩 선언 되기때문에.

    int grid[5][5] = {
        {0, 1, 2, 3, 4},
        {5, 6, 7, 8, 9},
        {10, 11, 12, 13, 14},
        {15, 16, 17, 18, 19},
        {20, 21, 22, 23, 24},
    };

    int src[5][5] = {
        {0},
        {5, 6, 7, 8, 9},
        {10, 11, 12, 13, 14},
        {15, 16, 17, 18, 19},
        {20, 21, 22, 23, 24},
    };
//  퀴즈: 그리드의 3번째 요소를 show_def 함수로 출력하세요.

    show_def(grid[0],25);
    int dst[25] = {0};
    // show_def(dst,25);

// 퀴즈: 3가지 방법으로 호출하세요.
    copy_1(grid, dst);
    // copy_1(grid[0], dst);
    // copy_1(&grid[0][0], dst);

    show_def(dst,25);

    copy_2(src, dst);

    // printf("\n%d\n",src[3][3]);

    show_def(src,25);


    


    // 퀴즈: 2차원 배열 grid를 1차원 배열 dst에 복사하는 함수를 만드세요.

}

void copy_1(int grid[], int dst[]){
    for (int i = 0; i < 25; i++){
        dst[i] = *grid + i;
    }
}

//  퀴즈: 1차원 배열 a와 b를출력 할 수 있는 함수를 만드세요.


void show_def(int a[], int size1){
    for(int i = 0; i < size1; i++){
        printf("%d ",a[i]);
    }
    printf("\n");
}

//  퀴즈: 1차원 배열을 2차원으로 만드는 함수를 만드세요.


void copy_2(int src[][5], int dst[]){
    for (int i = 0; i < 25; i++){
        src[0][i] = dst[i];
    }
}