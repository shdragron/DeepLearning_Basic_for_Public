// 11_5_enum.cpp

#include <iostream>
using namespace std;

// 추상 함수는 상속 받는 클래스에서 반드시 구현해야 한다.
class Shape {
    virtual void draw() = 0;
};

class Rectangle : public Shape {
public:
    int width, height;
    void draw(){
        std::cout << "사각형" << std::endl;
    };

};



class Circle : public Shape {
    public:
    int radius;
    void draw(){
        std::cout << "원" << std::endl;
    };
};



// enum은 상수의 집합을 정의할 때 사용한다.
enum Color { RED, GREEN, BLUE};
        //    0     1     2


int main() {
    Rectangle r;
    r.draw();
    Color c = Color::GREEN;
    switch (c) {
        case RED:
            std::cout << "RED" << std::endl;
            break;
        case GREEN:
            std::cout << "GREEN" << std::endl;
            break;
        case BLUE:
            std::cout << "BLUE" << std::endl;
            break;
    }
    return 0;
}