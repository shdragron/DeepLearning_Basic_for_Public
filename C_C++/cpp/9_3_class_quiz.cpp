// 9_2_class.cpp

#include <iostream>
#include <cstdio>

using namespace std; // 구분 의미 있는 이름을 주는 것이다. 이름 충돌을 방지 하기위해 사용.

class Point {

    // 아무것도 지정안하면 접근지정자가 
    // 1) private: 나만 사용가능(해당 클라스만 사용가능),
    // 2) public 누구나 가능, 
    // 3) protected: 나하고 사이 퍼블릭 사이 -> 상속한 class만 가능

public:

    int x,y;

    //constructor(c'tor)
    Point() { // 기본 생성자
        std::cout << "기본 생성자 호출" << endl;
        x = y = 0;
    }

    Point(int x, int y) { //생성자
        std::cout <<"생성자 호출" << endl;
        this->x = x; //this 는 자기 자신을 가리키는 주소 -> 파이썬의 self 
        this->y = y; //this 는 자기 자신을 가리키는 주소 ->를 해서 값을 나타낸다.
    }

    //destructor(d'tor)
    ~Point() { // 소멸자: 객체가 끝날때 실행되는 함수

        std::cout << "소멸자 호출" << endl;

    }

    void show() { // 매개변수 전달을 하지 않아도 가지고 있다. 

        std::cout << x << "," << y << endl;

    }

};


int main(){
    std::cout << "hello, c++!!" << " " << 1234 << endl;    // ::은 쓰겠다., cout은 콘솔아웃, << 타입이 달라서 컴파일러 선에서 타입을 추론한다.
    std::cout << true << false << endl; //Boolean 타입:

    // pt.x = 10; // 객체 지향적이지 않는 코드
    // pt.y = 20;
    // std::cout << pt.y << endl;

    Point pt2; // 다형성 -> 같은 이름의 함수를 매개변수로 판단.

    Point pt(3,7);  // 변수를 만들때 호출되는 함수인 생성자를 호출(비어 있어도.)

    /////////////////////////////////////////////////////////////////////////////////////////
    Point *p = new Point(10,20);    // 동적 할당 (malloc, free) -> 소멸자도 수동이다.
    p->show();
    delete p;   // 소멸자 수동실행
    p = NULL;   // 사용할 수 없다라고 가리킨다
    /////////////////////////////////////////////////////////////////////////////////////////

    pt2.show();
    pt.show();

    return 0;
}


// endl -> 개행 문자
// cout등의 매커니즘. 그리고 << 1234 호출
// operator <<(const char* s){
//     return cout;
// }

// 객체: 태어날때부터 완전하다. -> class 가 정의 된거 부터