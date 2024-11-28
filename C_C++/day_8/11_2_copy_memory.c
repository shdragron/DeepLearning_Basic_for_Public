// 11_2_copy_memory.c

#include <stdio.h>
#include <memory.h>

int main(){

    int a[5] = {1 ,3 ,5, 6, 9};
    char b[10] = "hollywood";
    double c[3] = {1.23, 4.56, 7.89};

    int aa[5] = {0};
    char bb[10] = "";
    double cc[3] = {0};

    memcpy(aa, a, sizeof(a));
    memcpy(bb, b, sizeof(b));
    memcpy(cc, c, sizeof(c));

    printf("aa: %d, %d, %d, %d, %d\n", aa[0], aa[1], aa[2], aa[3], aa[4]);
    printf("bb: %s\n", bb);
    printf("cc: %lf, %lf, %lf\n", cc[0], cc[1], cc[2]);


    return 0;
}