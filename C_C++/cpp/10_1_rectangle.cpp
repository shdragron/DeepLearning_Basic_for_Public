// 10_1_rectangle.cpp

#include <iostream>
#include <cmath>

using namespace std;

class Point {

    public: int x,
    y;

    Point() { // 기본 생성자
        std::cout << "기본 생성자 호출" << endl;
        x = y = 0;
    }

    Point(int x, int y) { //생성자
        std::cout << "생성자 호출" << endl;
        this - >x = x; //this 는 자기 자신을 가리키는 주소 -> 파이썬의 self
        this - >y = y; //this 는 자기 자신을 가리키는 주소 ->를 해서 값을 나타낸다.
    } ~ Point() { // 소멸자: 객체가 끝날때 실행되는 함수

        std::cout << "소멸자 호출" << endl;

    }

    void show() { // 매개변수 전달을 하지 않아도 가지고 있다.

        std::cout << x << "," << y << endl;

    }

};

class Reactangle {


    public:
    int x, y, x1, y1;

    // int x, y, x1, y1; Reactangle(int x = 0, int y = 0, int x1 = 0, int y1 = 0) {
    // 생성자 -> default 값이 있기 때문에 매개변수를 안넣어서 기본 생성자처럼 동작한다.     std::cout << "생성자 호출"
    // << endl;     this -> x = x;     this -> y = y;     this -> x1 = x1;     this
    // -> y1 = y1; }

    Reactangle(int x = 0, int y = 0, int x1 = 0, int y1 = 0) { //생성자 -> default 값이 있기 때문에 매개변수를 안넣어서 기본 생성자처럼 동작한다.

        std::cout << "생성자 호출" << endl;
        this ->x = x;
        this ->y = y;
        this ->x1 = x1;
        this ->y1 = y1;
    } ~ Reactangle() { // 소멸자: 객체가 끝날때 실행되는 함수

        std::cout << "소멸자 호출" << endl;

    }

    inline int area() { // 매개변수 전달을 하지 않아도 가지고 있다.
        return ((x1 - x) * (y1 - y));
    }

    inline double length() { // 매개변수 전달을 하지 않아도 가지고 있다.
        return sqrt(pow(x1 - x, 2) + pow(y1 - y, 2));
    }

    inline int height() { // 한줄은 inline은 함수를 호출하는 것이 아니라 함수의 내용을 호출한다. -> 더 빠르다.
        return y1 - y;
    }
};

// 퀴즈: point 클래스를 이용하여 Reactangle 클랙스를 만드세요. 변수: 좌표 2개 함수: 면적, 길이, 높이 출력

int main() {

    Point p(0, 0);
    Point p1(1, 1);
    Reactangle r(p.x, p.y, p1.x, p1.y);
    std::cout << p.x << p.y << p1.x << p1.y << endl;

    // std::cout << r.area() << endl; std::cout << r.height() << endl; std::cout <<
    // r.length() << endl;

    Reactangle basic;
    basic.area();
    return 0;
}