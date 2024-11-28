// 11_3_linkedlist.cpp

#include <iostream>

using namespace std;

struct NODE {
    int data;
    NODE * next;

    NODE(int n) : data(n) {
        next = NULL;
    }
};

void print_list(NODE* HEAD);

int main() {
    
    NODE n0(0);
    NODE n1(1);
    NODE n2(2);

    // 연결
    NODE* HEAD = NULL;
    HEAD = &n0;
    n0.next = &n1;
    n1.next = &n2; // tail 노드는 next가 NULL이다.

    // std::cout << n0.data << " " << (n0.next)->data << " " << ((n0.next)->next)->data << endl; // 왜 별이 안되고 -> 쓰는지 물어보자.

    std::cout << HEAD->data << " " << (*(HEAD->next)).data << " " << ((HEAD->next)->next)->data << endl; // 왜 별이 안되고 -> 쓰는지 물어보자.
    print_list(HEAD);

    return 0;
}

void print_list(NODE* HEAD) {
    for (NODE* p = HEAD; p != NULL; p = p->next) {

        std::cout << p->data << " ";

    }

    std::cout << endl;

}