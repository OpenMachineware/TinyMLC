// 自动生成的代码，请勿手动修改
// 由 tinymlc 自动生成

#ifndef TINYMLC_MODEL_H
#define TINYMLC_MODEL_H

#include <stdint.h>

#define INPUT_SIZE {{ input_size }}
#define OUTPUT_SIZE {{ output_size }}

// 推理函数声明
void {{ inference_func }}(const int8_t* input, int8_t* output);

#endif // TINYMLC_MODEL_H
