/**
 * NMSIS-NN accelerated softmax operator
 *
 * NMSIS-NN softmax API differs significantly from TinyMLC's tmlc_softmax_s8.
 * Using pure C implementation instead.
 */

#include "tinymlc.h"

// Use pure C softmax - NMSIS-NN API is incompatible
void tmlc_softmax_s8(const int8_t* input, int8_t* output, int size) {
    int8_t max = -128;
    for (int i = 0; i < size; i++) {
        if (input[i] > max) max = input[i];
    }

    int32_t sum = 0;
    int32_t exp_vals[size];
    for (int i = 0; i < size; i++) {
        int32_t x = (int32_t)(input[i] - max);
        int32_t exp_val = (x >= 0) ? (1 << (x / 4)) : (1 >> ((-x) / 4));
        exp_vals[i] = exp_val;
        sum += exp_val;
    }

    for (int i = 0; i < size; i++) {
        int32_t prob = (exp_vals[i] * 128) / sum;
        output[i] = (int8_t)prob;
    }
}