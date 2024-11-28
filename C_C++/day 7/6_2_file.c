// 6_2_file.c

#include <stdio.h>
#define TRUE 1
#define FALSE 0


int main(){
    FILE* fp = fopen("/Users/moongiyong/Desktop/NVIDIA AI ACADEMY/NVIDIA-AI-ACADEMY/C:C++/data/numbers.csv", "r");
    char buffer[256];

    // while(fgets(buffer,sizeof(buffer)),fp){
    //     printf("%s\n", buffer);
    // }

    int a;
    int c;
    float b;

    while (fscanf(fp, "%d%,%f,%d",&a, &b, &c) != EOF){
        printf("%d, %f, %d", a, b, c);
        
    }
    
    fclose(fp);
    return 0;
}