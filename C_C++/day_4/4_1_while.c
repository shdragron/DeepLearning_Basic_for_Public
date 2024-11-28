#include <stdio.h>

void f_2();

int main() {
    f_2();
    printf("%d",0x80000000);
    return 0;
}

// void f_1() {
//     int I = 0;
//     int result = 0;

//     while (1) {
//         scanf("%d", &I);
//         if (I == -1) {
//             printf("Error");
//             break;
//         }
//         else{
//             result += I; 
//             printf("입력받은 정수: %d\n", result); 
//         }
        
//     }

// }

// 퀴즈: 0을 입력할 때까지의 정수 중에서 가장 큰 값을 알려주세요.

void f_2(){
    int I = 0;
    int tmp = 0;
    while(1){
        printf("정수 입력: ");
        scanf("%d",&I);
        if(I==0){
            printf("가장큰값: %d",tmp);
            break;
        }
        else{
            if (tmp <= I){
                tmp = I;
            }
        }
    }
}