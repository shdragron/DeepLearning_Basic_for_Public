// 10_2_reference.cpp

#include <iostream>
using namespace std;

void swap(int& x, int& y);

int main(){
    // int value = 13;
    // int* ref = &value; // 포인터

    // *ref = 99;

    int value = 13;
    int ref = 12; //참조 변수 int& value랑 동일

    std::cout << ref << endl;
    std::cout << value << endl;

    swap(value,ref);

    std::cout << ref << endl;
    std::cout << value << endl;

    return 0;
}

// 퀴즈: 레퍼런스를 이용해서 a와 b를 교환하는 함수를 만드세요.

void swap(int& x, int& y){ // 참조변수는 함수안에서 전역 변수처럼 사용할 수 있다.
    int tmp = x;
    x = y;
    y = tmp;
}