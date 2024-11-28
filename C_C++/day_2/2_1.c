#include <stdio.h>
// void f_1();
// void f_3();
// void f_4();
// void f_5();
void f_6();

int main(){
    // f_1();
    // f_3();
    // f_4();
    // f_5();
    f_6();
}

// void f_1(){
//     printf("%d\n",7/3); //정수 나눗셈
//     printf("%.20lf\n",7.0/3.0); // 실수 나눗셈

//     int a =7, b =3;
//     printf("%d\n",a / b);
//     printf("%lf\n",a / (double) b); //형 변환
//     printf("%f\n",a / (float) b); //형 변환

//     printf("%d\n", 7 % 3); // 1

// }

// void f_3() {
//     double f = 0.5;
//     double b;
//     printf("%lf", 0.5);
//     scanf("%lf",&b); //-> 공백이 없어야됨.
//     printf("%lf", b);
// }
// 3 + 5.2

// 2_1. 퀴즈
//정수와 실수 변수를 각각 만들어서 sanf 한번만 써서 값을 입역 받으세요.

// void f_4() {
//     int a;
//     double b;
//     scanf("%d,%lf",&a,&b); //-> 공백이 없어야됨.
//     printf("%d\n%lf", a,b);
// }

// void f_5() {
//     //2_2. 퀴즈
//     //실수를 입력 받아서 소수 점이하만 남기는 코드를 만드세요.
//     double f = 4.5;
//     int x = f;
//     printf("%lf",(f - x));
//     //형변환 사용 -> 한줄로 가능!
//     printf("%lf", f - (int) f);
// }
void f_6(){
    // 퀴즈
    // 2자리 양수를 거꾸로 뒤집어 주세요.
    // int a = 35;
    // printf("%d\n", (a%10)*10 + a/10 );
    //퀴즈
    //3자라리 양수를 거꾸로
    // int c = 256;

    // printf("%d", (c / 100) + ((c / 10)-(c / 100) * 10) * 10 + (c % 10) * 100);
    
    //퀴즈
    //a를 b로 나눈 나머지를 %연간자 없이 구하세요.
    int a = 7, b =3;
    printf("%d\n" , a - b * (a/b));

    //배열
    int ary[4] = {1,2,3,4};
    printf("%d",ary[2]);

}  