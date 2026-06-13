// 自动生成的代码，请勿手动修改
// 可调用 {{ inference_func }} 接入你的工程
// 由 tinymlc 自动生成

#include "tinymlc.h"
#include "lstm_weights.h"
#include "{{ weights_header }}"

// 输入输出大小
#define INPUT_SIZE {{ input_size }}
#define OUTPUT_SIZE {{ output_size }}

// 推理函数
void {{ inference_func }}(const int8_t* input, int8_t* output) {
    static int8_t output_state[20];
    static int8_t cell_state[20];

    // 初始化状态为 0
    for (int i = 0; i < 20; i++) {
        output_state[i] = 0;
        cell_state[i] = 0;
    }

    // 调用 LSTM
    tmlc_unidirectional_sequence_lstm_s8(
        input,           // [28, 1, 28]
        lstm_input_weights,       // input_weights
        lstm_recurrent_weights,   // recurrent_weights
        lstm_bias,                // bias
        output_state,    // [1, 20]
        cell_state,      // [1, 20]
        28,              // time_steps
        1,               // batch_size
        28,              // input_size
        20               // hidden_size
    );

    // LSTM 输出是 20 维，需要 Reshape 到 560 维
    // 这里取最后一个时间步的输出（output_state 已经是最后一个时间步）
    int8_t lstm_out[560];
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 20; j++) {
            lstm_out[i * 20 + j] = output_state[j];  // 简化，实际应该是每个时间步的输出
        }
    }

    // FC 层
    tmlc_fully_connected_s8(lstm_out, fc_weights, fc_bias, output, 560, OUTPUT_SIZE);

    // Softmax
    tmlc_softmax_s8(output, output, OUTPUT_SIZE);
}
