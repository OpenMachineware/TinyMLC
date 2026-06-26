#include "tinymlc.h"

// SVDF internal clip function (not real tanh, just clip to int8 range)
static int8_t svdf_clip(int32_t x) {
    const int32_t max_val = 127;
    const int32_t min_val = -128;
    if (x > max_val) return max_val;
    if (x < min_val) return min_val;
    return (int8_t)x;
}

void tmlc_svdf_s8(const int8_t* input,
                  const int8_t* weights,
                  const int32_t* bias,
                  int8_t* output,
                  int time_steps,
                  int input_size,
                  int rank,
                  int units)
{
    int output_size = rank * units;
    
    for (int t = 0; t < time_steps; t++) {
        const int8_t* input_ptr = input + t * input_size;
        int8_t* output_ptr = output + t * output_size;
        
        for (int i = 0; i < output_size; i++) {
            int32_t sum = bias[i];
            for (int j = 0; j < input_size; j++) {
                int32_t w_val = (int32_t)weights[i * input_size + j];
                sum += (int32_t)input_ptr[j] * w_val;
            }
            output_ptr[i] = svdf_clip(sum >> 8);
        }
    }
}