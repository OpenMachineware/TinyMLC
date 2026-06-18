#include "tinymlc.h"

void tmlc_relu_s8(const int8_t* input, int8_t* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] < 0 ? 0 : input[i];
    }
}
