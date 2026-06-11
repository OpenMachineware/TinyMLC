// 自动生成的代码，请勿手动修改
// 由 tinymlc 自动生成

#include <stdint.h>
#include "tinymlc.h"
#include "{{ weights_header }}"

// 输入输出大小
#define INPUT_SIZE {{ input_size }}
#define OUTPUT_SIZE {{ output_size }}

// 推理函数
void {{ inference_func }}(const int8_t* input, int8_t* output) {
    // LSTM 输出缓冲区 (28 * 20 = 560)
    int8_t lstm_output[560];

    // TODO: 实现 UNIDIRECTIONAL_SEQUENCE_LSTM
    // 暂时直接 reshape 输入作为占位
    for (int i = 0; i < 560; i++) {
        lstm_output[i] = input[i % INPUT_SIZE];
    }

    // FC 层
    tmlc_fully_connected_s8(lstm_output, fc_weights, fc_bias, output, 560, OUTPUT_SIZE);

    // Softmax
    tmlc_softmax_s8(output, output, OUTPUT_SIZE);
}
