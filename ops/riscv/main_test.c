// RISC-V test entry
// Uses model.h for INPUT_SIZE and OUTPUT_SIZE definitions

#include <stdint.h>
#include "model.h"
#include "debug_print.h"

#define SIFIVE_TEST_ADDR ((volatile uint32_t*)0x100000)

int main() {
    // Test inputs - initialized with constant value
    static int8_t input1[INPUT_SIZE_1];
    static int8_t output[OUTPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE_1; i++) {
        input1[i] = 1;
    }

    // Call inference
#if defined(INPUT_SIZE_2) && INPUT_SIZE_2 > 0
    static int8_t input2[INPUT_SIZE_2];
    for (int i = 0; i < INPUT_SIZE_2; i++) {
        input2[i] = 1;
    }
    tinymlc_inference(input1, input2, output);
#else
    tinymlc_inference(input1, output);
#endif

    // Output result
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        TMLC_PRINT_INT(output[i]);
        TMLC_PUTCHAR(' ');
    }
    TMLC_PUTCHAR('\n');

    DEBUG_STR("ALL-OK\n");

    *SIFIVE_TEST_ADDR = 0x3333;  // QEMU auto exit

    return 0;
}
