// 自动生成的代码，请勿手动修改
// 可调用 {{ inference_func }} 接入你的工程
// 由 tinymlc 自动生成

#include "tinymlc.h"
{{ includes }}

// 输入输出大小
#define INPUT_SIZE {{ input_size }}
#define OUTPUT_SIZE {{ output_size }}

{% if has_lstm %}
// LSTM 参数（从模型动态获取）
#define LSTM_TIME_STEPS {{ lstm_time_steps }}
#define LSTM_BATCH_SIZE {{ lstm_batch_size }}
#define LSTM_INPUT_SIZE {{ lstm_input_size }}
#define LSTM_HIDDEN_SIZE {{ lstm_hidden_size }}

// 推理函数
void {{ inference_func }}(const int8_t* input, int8_t* output) {
    static int8_t output_state[LSTM_HIDDEN_SIZE];
    static int8_t cell_state[LSTM_HIDDEN_SIZE];

    // 初始化状态为 0
    for (int i = 0; i < LSTM_HIDDEN_SIZE; i++) {
        output_state[i] = 0;
        cell_state[i] = 0;
    }

    // 调用 LSTM
    tmlc_unidirectional_sequence_lstm_s8(
        input,
        lstm_input_weights,
        lstm_recurrent_weights,
        lstm_bias,
        output_state,
        cell_state,
        LSTM_TIME_STEPS,
        LSTM_BATCH_SIZE,
        LSTM_INPUT_SIZE,
        LSTM_HIDDEN_SIZE
    );

    // Reshape: [LSTM_HIDDEN_SIZE] -> [LSTM_TIME_STEPS * LSTM_HIDDEN_SIZE]
    int8_t lstm_out[LSTM_TIME_STEPS * LSTM_HIDDEN_SIZE];
    for (int i = 0; i < LSTM_TIME_STEPS; i++) {
        for (int j = 0; j < LSTM_HIDDEN_SIZE; j++) {
            // FIXME 简化，实际应该是每个时间步的输出
            lstm_out[i * LSTM_HIDDEN_SIZE + j] = output_state[j];
        }
    }

    {% if has_fc %}
    // FC 层
    tmlc_fully_connected_s8(lstm_out, fc_weights, fc_bias, output,
                            LSTM_TIME_STEPS * LSTM_HIDDEN_SIZE, OUTPUT_SIZE);
    {% endif %}

    // Softmax
    tmlc_softmax_s8(output, output, OUTPUT_SIZE);
}
{% else %}
// 没有 LSTM，只有 FC 的简化版本
void {{ inference_func }}(const int8_t* input, int8_t* output) {
    // 直接 FC
    tmlc_fully_connected_s8(input, fc_weights, fc_bias, output, INPUT_SIZE, OUTPUT_SIZE);
    tmlc_softmax_s8(output, output, OUTPUT_SIZE);
}
{% endif %}
