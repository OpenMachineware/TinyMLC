// 自动生成的代码，请勿手动修改
// 由 tinymlc 自动生成

#ifndef TINYMLC_MODEL_H
#define TINYMLC_MODEL_H

#include <stdint.h>

#define INPUT_SIZE {{ input_size }}
#define OUTPUT_SIZE {{ output_size }}

{% if has_lstm %}
// LSTM 右移位数（从模型量化参数计算）
#define LSTM_SHIFT_I {{ lstm_shifts[0] }}
#define LSTM_SHIFT_F {{ lstm_shifts[1] }}
#define LSTM_SHIFT_G {{ lstm_shifts[2] }}
#define LSTM_SHIFT_O {{ lstm_shifts[3] }}
{% endif %}

// 推理函数声明
void {{ inference_func }}(const int8_t* input, int8_t* output);

#endif // TINYMLC_MODEL_H
