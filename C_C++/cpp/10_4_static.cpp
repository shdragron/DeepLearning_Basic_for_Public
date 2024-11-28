// 10_4_static.cpp
#include <iostream>
#include <cstring>

using namespace std;

//  퀴즈: 프로그램 끝날 때까지 만든 Product 객체는 몇개인지 알려주세요.
// int g_product_count = 0;

class Product {
    private:

        static int count; // static 변수는 전역변수와 비슷하다. -> 정적 멤버 변수

    public: 

        char name[256];
        int price;

        Product(const char * name, int price) {
            strcpy(this -> name, name );
            this -> price = price;
            count++;
        }

        void show() {

            std::cout << name << "," << price << endl;
        }
        
        static void show_count() { // static 함수는 static 변수만 사용할 수 있다. -> 객체 변수 사용 불가능
            std::cout << "생성된 객체 수: " << count << endl;
        }
};

int Product::count = 0;

int main() {
    Product p1("apple", 1000);
    Product p2("pineapple", 1000);
    Product p3("pear", 1000);

    p1.show();
    // std::cout << "생성된 객체 수: " << g_product_count << endl;
    p1.show_count();
    // p3.show_count(); -> p1, p2, p3 모두 같은 count를 가지고 있기 때문에 p1을 호출해도 된다.
    Product::show_count(); // -> 이게 정석적인 방법이다.

    return 0;
}