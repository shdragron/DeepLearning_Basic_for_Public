// 11_5_single_list.cpp

#include <iostream>
using namespace std;

struct NODE {
    int data;
    NODE* next;

    NODE(int n) : data(n) { 
        next = NULL;
    }
};

class SingleList {
private:
    NODE* head;

public:
    SingleList() : head(NULL) {}
    ~SingleList() { clear(); }

    bool is_empty() { return head == NULL; }

    void show_all() {
        NODE* cur = head;
        while (cur != NULL) {
            std::cout << cur->data << ", ";
            cur = cur->next;
        }
        std::cout << endl;
    }

    void add_head(int value) {
        NODE* node = new NODE(value);
        node->next = head;
        head = node;
    }

    void add_tail(int value) {
        // NODE* node = new NODE(value);
        // if (head == NULL) {
        //     head = node;
        //     return;
        // }

        // NODE* cur = head;
        // while (cur->next != NULL)
        //     cur = cur->next;
        // cur->next = node;

        NODE** cur = &head;
        while (*cur)
            cur = &(*cur)->next;
        *cur = new NODE(value);

        // int a = 7;
        // int* p;
        // p = &a;
        // *p = 9;
    }

    int remove_head() {
        NODE* node = head;
        head = head->next; 

        int value = node->data;
        delete node;
        return value;
    }

    void clear() {
        while (is_empty() == false)
            remove_head();
    }
};

class Stack : private SingleList {
public:
    bool is_empty() { return SingleList::is_empty(); }
    void push(int n) { add_head(n); }
    int pop() { return remove_head(); }
    void show() { show_all(); }
};

class Queue : private SingleList {
public:
    bool is_empty() { return SingleList::is_empty(); }
    void enqueue(int n) { add_tail(n); }
    int dequeue() { return remove_head(); }
    void show() { show_all(); }
};

int main() {
    SingleList list;

    list.add_head(11);
    list.add_head(22);
    list.add_head(33);

    list.add_tail(55);
    list.add_tail(66);

    list.show_all();

    std::cout << list.remove_head() << ' ';
    std::cout << list.remove_head() << ' ';
    std::cout << list.remove_head() << endl;
    list.show_all();

    Stack s;
    for (int i = 0; i < 5; i++)
        s.push(i);

    s.show();
    while (s.is_empty() == false)
        std::cout << s.pop() << endl;

    return 0;
}


