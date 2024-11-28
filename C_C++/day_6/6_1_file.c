#include <stdio.h>
#include <string.h>
// void read_file();
void write_file();

int main(){
    // read_file();
    write_file();
    return 0;
}
// // EOF: End of file 파일을 EOF까지 읽으면 다시 열어야 돌아 갈 수 있다.
// void read_file(){
//     char file_path[] = "/Users/moongiyong/Desktop/NVIDIA AI ACADEMY/NVIDIA-AI-ACADEMY/C:C++/day_6/poem.txt";
//     char buffer[256];

//     FILE* fp = fopen(file_path,"r");
//     while(feof(fp)==0){ //파일의 끝에 도달했을때

//         fgets(buffer, sizeof(buffer),fp); // 한줄을 버퍼에 넣기 (넣는 변수, 크기, 스트링)
//         puts(buffer);
//     }
//     fclose(fp);
// }

// void read_file(){
//     char file_path[] = "/Users/moongiyong/Desktop/NVIDIA AI ACADEMY/NVIDIA-AI-ACADEMY/C:C++/day_6/poem.txt";
//     char buffer[256];

//     FILE* fp = fopen(file_path,"r");
//     while(feof(fp)==0){ //파일의 끝에 도달했을때

//         // fgets(buffer, sizeof(buffer),fp); // 한줄을 버퍼에 넣기 (넣는 변수, 크기, 스트링) -> 성공하면 해당줄 첫 포인터, 실패하면 null의 포인터를 줌.
//                                           // NULL 포인터: 주소가 0인 포인터. -> 가리키는 곳이없다.
//         // buffer[strlen(buffer)-1] = '\0';
//         if(buffer[scanf(buffer)-1] =='\n'){

//         }
//         else{
//             puts(buffer); //화먄에 츨력

//         }
//         // buffer[0] = '\0'; // 버퍼 클리어

        
//     }
//     fclose(fp);
// }

// void read_file(){
//     FILE* fp = fopen(file_path,"r");
//     char file_path[] = "/Users/moongiyong/Desktop/NVIDIA AI ACADEMY/NVIDIA-AI-ACADEMY/C:C++/day_6/poem.txt";
//     char buffer[256];
//     int ch;

//     FILE* fp = fopen(file_path,"r");

//     while(1){
//         ch = fgetc(fp);
//         printf("%d\n", ch);

//         if(ch == EOF){ // -1
//             break; 
//         }
//         putchar(ch);
//     }
// }

void write_file(){

    FILE* fp = fopen("/Users/moongiyong/Desktop/NVIDIA AI ACADEMY/NVIDIA-AI-ACADEMY/C:C++/day_6/sample.txt","a");

    if(fp == NULL){
        return;
    }
    // fputs("아 행복하다.\n", fp);
    fprintf(fp,"이름: %s, 학번: %d", "기용",2020094384); // 이게 가작 중요!!

    fclose(fp);

}