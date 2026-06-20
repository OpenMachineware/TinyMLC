#include <stdint.h>

extern void uart_init(void);
extern void tinymlc_putchar(char c);
extern void tinymlc_print_int(int n);
extern void tinymlc_inference(const int8_t* input, int8_t* output);

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define SIFIVE_TEST_ADDR ((volatile uint32_t*)0x100000)

void qemu_exit(int exit_code) {
    uint32_t params[2] = {0x20, (uint32_t)exit_code};
    __asm__ volatile(
        "mov r0, #0x18\n"
        "mov r1, %[p]\n"
        "bkpt #0xab\n"
        :
        : [p] "r"(params)
        : "r0", "r1", "memory");
    while (1);
}

int main() {
    uart_init();

    static int8_t input[INPUT_SIZE];
    static int8_t output[OUTPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = 1;
    }

    tinymlc_inference(input, output);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        tinymlc_print_int(output[i]);
        tinymlc_putchar(' ');
    }
    tinymlc_putchar('\n');

    qemu_exit(0);
    return 0;
}
