#include <stdio.h>
#include <string.h>

int sum(int n);
void pr(char* c, int size);

int main(){
    printf("%d\n", sum(10));
    char c[256] = "hello";
    pr(c, strlen(c));

    return 0;
}

// sum(10): 1부터 10까지 합게: sum(9) + 10
// sum(9): 1부터 9까지의 합개: sum(8) + 9
// sum(8): sum(7) + 8
// ...
// sum(1): 1

int sum(int n){
    if (n==1)
        return 1;
    return sum(n-1) + n;
}

// 퀴즈: 문자열을 거꾸로 출력하는 재귀함수를 만드세요.

void pr(char* c, int size){
    if (size == 0){
        return;
    }
    putchar(c[size-1]);
    pr(c, size-1);
}
