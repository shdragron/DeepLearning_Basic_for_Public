// 10_3_overloading.cpp
// 기본생성자, 대입 연산자, 복사 생성자는 기본으로 만들어져 있다.
// 오버로딩 구현 -> 1) class내부, 2) 외부

#include <iostream>

using namespace std;

class Point {

public:

    int x,y;


    Point() { // 기본 생성자
        x = y = 0;
        std::cout << "기본 생성자 호출" << endl;
    }

    Point(int x, int y) { //생성자
        std::cout << "좌표 생성자" << endl;
        this->x = x; //this 는 자기 자신을 가리키는 주소 -> 파이썬의 self 
        this->y = y; //this 는 자기 자신을 가리키는 주소 ->를 해서 값을 나타낸다.
    }

    Point(const Point& other) { //복사 생성자 copy constructor
        std::cout << "복사 생성자" << endl;
        this->x = other.x; //this 는 자기 자신을 가리키는 주소 -> 파이썬의 self 
        this->y = other.y; //this 는 자기 자신을 가리키는 주소 ->를 해서 값을 나타낸다.
    
    }

    Point& (operator =)(Point other){ // 얕은 복사  this 를 return하면 함수 끝나도 살아있기때문에 Point&를 사용한다.
        std::cout << "대입 연산자" << endl;
        this->x = other.x;
        this->y = other.y;
        return *this;   //other도 가능
    }

    Point add(const Point rhs) { // Point인 이유는 Point(a,b)는 함수가 나면 사라지기 때문에 새로 만들었다.
        int a = this->x + rhs.x;
        int b = this->y + rhs.y;
        return Point(a,b);
    }

    ~Point() { // 소멸자: 객체가 끝날때 실행되는 함수

    }

    void show() { // 매개변수 전달을 하지 않아도 가지고 있다. 

        std::cout << x << "," << y << endl;

    }

    // 1) class안 오버로딩

    // Point (operator +)(const Point rhs) { // 넘어 온 값은 변화하지 않기때문에  const를 사용해야된다.
    //     int a = this->x + rhs.x;
    //     int b = this->y + rhs.y;
    //     return Point(a,b);
    // }

    // 3)외부애서 오버로딩된 것을 사용하려면 friend를 사용해야한다.

    friend Point (operator +)(const Point rhs, const Point lhs) { // 넘어 온 값은 변화하지 않기때문에  const를 사용해야된다.
            int a = lhs.x + rhs.x;
            int b = lhs.y + rhs.y;
            return Point(a,b);
};

};

    // 2) class 밖 오버로딩

Point (operator +)(const Point rhs, const Point lhs) { // 넘어 온 값은 변화하지 않기때문에  const를 사용해야된다.
        int a = lhs.x + rhs.x;
        int b = lhs.y + rhs.y;
        return Point(a,b);
};



int main() {
    Point pt1;
    Point pt2(3, 7);
    Point pt3 = pt2; // pt3는 **복사** 생성자를 호출해서 pt2를 복사한다. (얕은 복사)
    Point pt4(pt2); // pt4는 **복사** 생성자를 호출해서 pt2를 복사한다.(얕은 복사)

    pt3.show();
    pt4.show();

    pt1 = pt2; // pt1에 pt2를 대입한다 -> 복사 생성자는 아니다. -> 대입연산자 오버로딩을 해야한다. (얕은 복사)
    pt1 = pt2; // pt1에 대한 메소드로 pt2(others)를 대입한다.


    pt1.show();

    Point pt5 = pt1+pt2; // pt1.operator+(pt2) -> pt1 + pt2
    // Point pt6 = add_1(pt1, pt2); // pt1.operator+(pt2) -> pt1 + pt2

    pt1.show();
    pt5.show();
    // pt6.show();


    return 0;
}