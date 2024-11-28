// 7_2_2d_array.c

#include <stdio.h>

void show_grid(int grid[][6]); //포인터
void show_shape(int grid[5][5]); //포인터
void pick(int grid[5][5],int c);


int main(){

    int grid[5][5] = {
        {0, 1, 2, 3, 4},
        {5, 6, 7, 8, 9},
        {10, 11, 12, 13, 14},
        {15, 16, 17, 18, 19},
        {20, 21, 22, 23, 24},
    }; //int [5] (정수)4바이트 * 5 = 20바이트 *4 = 80 바이트

    printf("\n");


    // show_shape(grid);
    // pick(grid,2);
    show_grid(grid);
    return 0;
}

void show_grid(int grid[][6]){ // 포인터 
    for(int i = 0;i <= 4;i++){
        for(int j = 0; j <= 4; j++){
            if(grid[i][j] < 10){
                printf("0%d ",grid[i][j]);
            }
            else{
                printf("%d ",grid[i][j]);
            }
        }
        printf("\n");
    }
    printf("\n");
    
}










// 퀴즈 아래처럼 배열을 출력하세요.
// 파이썬

void show_shape(int grid[5][5]){ // 포인터
    for(int a = 0;a <= 4;a++){
        for(int b = 0; b <= 4; b++){
            if(grid[a][b]<10){
                if(a>=b){
                    printf("%2d ",grid[a][b]);
                }
                else{
                    printf("-- ");
                }
            }
            else{
                if(a>=b){
                    printf("%d ",grid[a][b]);
                }
                else{
                    printf("-- ");
                }
            }
        }
        printf("\n");
    }
    printf("\n");


    for(int a = 0;a <= 4;a++){
        for(int b = 0; b <= 4; b++){
            if(grid[a][b]<10){
                if(a<=b){
                    printf("%2d ",grid[a][b]);
                }
                else{
                    printf("-- ");
                }
            }
            else{
                if(a<=b){
                    printf("%d ",grid[a][b]);
                }
                else{
                    printf("-- ");
                }
            }
        }
        printf("\n");
    }
    printf("\n");


    for(int a = 0;a <= 4;a++){
        for(int b = 0; b <= 4; b++){
            if(grid[a][b]<10){
                if(a+b<=4){
                    printf("%2d ",grid[a][b]);
                }
                else{
                    printf("-- ");
                }
            }
            else{
                if(a+b<=4){
                    printf("%d ",grid[a][b]);
                }
                else{
                    printf("-- ");
                }
            }
        }
        printf("\n");
    }
    printf("\n");


    for(int a = 0;a <= 4;a++){
        for(int b = 0; b <= 4; b++){
            if(grid[a][b]<10){
                if(a+b>=4){
                    printf("%2d ",grid[a][b]);
                }
                else{
                    printf("-- ");
                }
            }
            else{
                if(a+b>=4){
                    printf("%d ",grid[a][b]);
                }
                else{
                    printf("-- ");
                }
            }
        }
        printf("\n");
    }
    printf("\n");

}

// 퀴즈 2차원 배열로 부터 특정한 열을 출력

void pick(int grid[5][5],int c){

    // for(int a = 0;a <= 4;a++){
    //     for(int b = 0; b <= 4; b++){
    //         if(b==c){
    //             printf("%d",grid[a][c]);
    //             break;
    //         }
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    for(int a = 0;a <= 4;a++){
        printf("%d",grid[a][c]);
        printf("\n");
    }
    printf("\n");
}