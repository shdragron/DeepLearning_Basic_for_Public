// 11_1_queue.cpp

#include <iostream>

using namespace std;

// - 3 5 6 7 - -
//   F     
//         R

// - 3 5 6 7
//   F     
// R

// -> 환형 큐


class Queue {
    public:
        static const int size = 44;
        // int buf[5] = {1, 3, 5, 6, 9};
        // int front, rear;
    private:

        int buf[size];
        int front, rear;
        
    public:
        Queue() : front(0), rear(0) { //int도 생성자가 있다고 생각해도 된다.
            std::cout << "생성자 호출"<< endl;
            // this->fornt == this->rear == 0;
    }
    // 퀴즈: 데이터를 추가하는 함수를 만드세요.

    void in_queue(int data) {

        this->buf[this->rear] = data;
        this -> rear = (this->rear + 1) % Queue::size;
        
    }

    // 퀴즈: 데이터를 삭제하는 함수를 만드세요.
    int out_queue() {

        this->front = (this->front + 2) % Queue::size;
        int tmp = this->buf[this->front];
        return tmp;

    }

    // 퀴즈: 비어있는지 확인하는 함수를 만드세요.
    bool is_empty() {

        return this->front == this->rear;

    }

    void show() {

        for (int i = this->front; i != this->rear; i = (i + 1) % Queue::size) {
                std::cout << buf[i] << ' ';
            }

            std::cout << endl;

        }
        // if (this->front < this->rear) {
        //     for (int i = this->front; i <= this->rear; i++) {
        //         std::cout << buf[i] << ' ';
        //     }
        //     std::cout << endl;
        // }

        // else if(this->front > this->rear) {
        //     for (int i = this->front; i < 5; i++) {
        //         std::cout << buf[i] << ' ';
        //     }

        //     for (int i = 0; i <= this->rear; i++) {
        //         std::cout << buf[i] << ' ';
        //     }
        //     std::cout << endl;
        // }

        // std::cout << buf[this->front] << ',' << buf[this->rear] << endl;
};

int main(){

    Queue q;

    for (int i = 1; i < 44; i++) {
        q.in_queue(i);
        q.show();
    }

    q.show();

    for (int j = 0; j < 41; j++) {
        q.in_queue(q.out_queue());
        q.in_queue(q.out_queue());
        q.out_queue();
    }

    std::cout << "마지막에 남은 두 사람은 " << q.out_queue() << "번째, " << q.out_queue() << "번째 사람입니다." << endl;
    q.show();
    
    return 0;
}

// 퀴즈: 43명이 둥글게 앉아있을 때, 2명이 남을 때까지 3번째 사람을 제거한다면 마지막에 남는 두 사람은 몇번째 사람일까요?