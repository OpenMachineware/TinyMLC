/**
 * ReLU activation operator
 *
 * CMSIS-NN uses activation parameters in conv/fc/pool structures.
 * For standalone ReLU, we use a simple implementation.
 */

#include "tinymlc.h"

// ReLU: y = max(0, x)
// For int8: y = max(-128, x) but clip at 0 for standard ReLU
void tmlc_relu_s8(const int8_t* input, int8_t* output, int size)
{
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}
