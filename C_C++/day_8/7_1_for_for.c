// 7_1_for_for.c

#include <stdio.h>

// 문제
// *   *
//  * *
//   *
//  * *
// *   *

// 문제
// *****
// *   *
// *   *
// *   *
// *****


int main(){
    int i = 0;
    int j = 0;
    // for(i = 0;i <= 4;i++){
    //     for(j = 0; j <= 4; j++){
    //         if(j==i && i+j==4){
    //             printf("*");
    //         }
    //         else if (i+j==4)
    //         {
    //             printf("*");   
    //         }
    //         else{
    //             printf("-");
    //         }
    //     }
    //     printf("\n");
    // }

    for(i = 0;i <= 4;i++){
        for(j = 0; j <= 4; j++){
            if(i > 0 && i < 4 && j > 0 && j < 4){
                printf("-");
            }
            else {
                printf("*");
            }
        }
        printf("\n");
    }
}
