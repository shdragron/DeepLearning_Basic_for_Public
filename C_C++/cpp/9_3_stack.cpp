// 9_3_stack.cpp

#include <iostream>
#include <cstdio>

using namespace std; // 구분 의미 있는 이름을 주는 것이다. 이름 충돌을 방지 하기위해 사용.


void push(int value);
int pop(int value);
int is_empty(); // 비어있는지 확인
void show_stack(int* value);

class Stack {
public:

    int g_buffer[256]; // 프로젝트 전체에 사용하는 변수 : 전역변수 g 글로벌
    int g_top = 0; // 데이터를 넣을 위치 [인덱스]

    void push(int value){
        //스택의 push 과정
        g_buffer[g_top++] = value;
    }

    int pop(int value){
        return g_buffer[--g_top]; // top 아래쪽이 데이터니까 -1을 해주고(데이터 받기) top도 낮춰줘야하니까 결론은 --g_top;
    }

    int is_empty(){
        return g_top == 0;
    }

    void show_stack(int* value){ // 알려주는 디버그 코드
        for (int i = g_top - 1; i >= 0; i--){
            printf("%d\n",value[i]);
        }
    }

};

int main(){

    // 퀴즈: 앞에서 만든 스택을 클래스 버전으로 업그레이드 하세여

    // 퀴즈: 스택을 사용해서 ages 배열을 거꾸로 뒤집어 주세요.

    Stack stack;

    int ages[5] = {12, 78, 45, 90, 34};

    for ( int i = 4; i >= 0; i--){ //g_buffer 리스트에 거꾸로된 함수 추가
        stack.push(ages[i]);
    }


    for ( int i = 0; i < 5; i++){ // ages에 덮어 씌우기
        ages[i] = stack.g_buffer[i];
        printf("%d\n",ages[i]);
    }



    return 0;
}