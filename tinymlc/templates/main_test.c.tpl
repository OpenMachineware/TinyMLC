// 自动生成的测试入口，请勿手动修改
// 由 tinymlc 自动生成

#include <stdint.h>
#include "{{ model_header }}"
#include "debug_print.h"


// UART 输出（用于测试）
#define UART_TX_ADDR ((volatile char*)0x10000000)
#define SIFIVE_TEST_ADDR ((volatile uint32_t*)0x100000)

static void putchar(char c) {
    *UART_TX_ADDR = c;
}

static void print_int(int n) {
    if (n < 0) {
        putchar('-');
        n = -n;
    }
    if (n >= 10) print_int(n / 10);
    putchar('0' + (n % 10));
}

int main() {
    int8_t input[INPUT_SIZE];
    int8_t output[OUTPUT_SIZE];

    // 测试输入：全 1
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = 1;
    }
    // 最后一行改为特殊值
    for (int i = (28-1)*28; i < 28*28; i++) {
        input[i] = 101;
    }

    // 调用推理
    {{ inference_func }}(input, output);

    // 输出结果
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        print_int(output[i]);
        putchar(' ');
    }
    putchar('\n');

    DEBUG_STR("ALL-OK\n");

    *SIFIVE_TEST_ADDR = 0x3333;  // QEMU 自动退出

    return 0;
}
