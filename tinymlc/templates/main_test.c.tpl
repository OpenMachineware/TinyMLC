// 自动生成的测试入口，请勿手动修改
// 由 tinymlc 自动生成

#include <stdint.h>
#include "{{ model_header }}"
#include "debug_print.h"

#define SIFIVE_TEST_ADDR ((volatile uint32_t*)0x100000)

int main() {
volatile char* uart = (volatile char*)0x09000000;  // ARM virt 板 UART
    uart[0] = 'S';
    uart[0] = 'T';
    uart[0] = 'A';
    uart[0] = 'R';
    uart[0] = 'T';
    uart[0] = '\n';

    static int8_t output[OUTPUT_SIZE];

    {% if inputs_count == 1 %}
        // 测试输入
        static int8_t input[INPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; i++) {
            input[i] = 1;
        }
        // 调用推理
        {{ inference_func }}(input, output);
    {% elif inputs_count == 2 %}
        static int8_t input1[{{ INPUT_SIZE_1 }}];
        static int8_t input2[{{ INPUT_SIZE_2 }}];
        for (int i = 0; i < {{ INPUT_SIZE_1 }}; i++) {
            input1[i] = 1;
        }
        for (int i = 0; i < {{ INPUT_SIZE_2 }}; i++) {
            input2[i] = 1;
        }
        // 调用推理
        {{ inference_func }}(input1, input2, output);
    {% endif %}

    // 输出结果
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        tinymlc_print_int(output[i]);
        tinymlc_putchar(' ');
    }
    tinymlc_putchar('\n');

    DEBUG_STR("ALL-OK\n");

    *SIFIVE_TEST_ADDR = 0x3333;  // QEMU 自动退出

    return 0;
}
