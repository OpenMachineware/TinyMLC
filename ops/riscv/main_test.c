// RISC-V test entry
// Uses model.h for INPUT_SIZE and OUTPUT_SIZE definitions

#include <stdint.h>
#include "model.h"
#include "debug_print.h"

// SiFive Test device address for QEMU exit
#define SIFIVE_TEST_ADDR ((volatile uint32_t*)0x100000)

// QEMU exit function for RISC-V virt machine
void qemu_exit(int exit_code) {
    *SIFIVE_TEST_ADDR = (exit_code << 16) | 0x3333;
    while (1);
}

int main() {
    // Test input - initialized with constant value
    static int8_t input[INPUT_SIZE];
    static int8_t output[OUTPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = 1;
    }

    // Call inference
    tinymlc_inference(input, output);

    // Output result
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        TMLC_PRINT_INT(output[i]);
        TMLC_PUTCHAR(' ');
    }
    TMLC_PUTCHAR('\n');

    DEBUG_STR("ALL-OK\n");

    qemu_exit(0);
    return 0;
}