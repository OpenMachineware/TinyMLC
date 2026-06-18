#include "tinymlc.h"

void tmlc_svdf_s8(const int8_t* input,
                  const int8_t* weights,
                  const int32_t* bias,
                  int8_t* output,
                  int time_steps,
                  int input_size,
                  int rank,
                  int units)
{
    // 简化实现
    for (int i = 0; i < units; i++) {
        output[i] = input[i % input_size];
    }
}
