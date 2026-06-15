// 自动生成的代码，请勿手动修改
// 可调用 {{ inference_func }} 接入你的工程
// 由 tinymlc 自动生成

#include "tinymlc.h"
{{ includes }}
#include "model.h"

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
    // LSTM 输出：完整序列 [time_steps, hidden_size]
    int8_t lstm_sequence[LSTM_TIME_STEPS * LSTM_HIDDEN_SIZE];

    // 可选：最后状态（当前不需要，传 NULL）
    int8_t last_state[LSTM_HIDDEN_SIZE];
    int8_t last_cell[LSTM_HIDDEN_SIZE];

    // 调用 LSTM（输出完整序列）
    tmlc_unidirectional_sequence_lstm_s8(
        input,
        lstm_input_weights,
        lstm_recurrent_weights,
        lstm_bias,
        lstm_sequence,   // 完整序列输出
        last_state,      // 最后状态（不需要）
        last_cell,       // 最后细胞状态（不需要）
        LSTM_TIME_STEPS,
        LSTM_BATCH_SIZE,
        LSTM_INPUT_SIZE,
        LSTM_HIDDEN_SIZE
    );

    // lstm_sequence 已经是 [time_steps * hidden_size] 的连续内存
    // 直接传给 FC 层，无需 Reshape

    {% if has_fc %}
    // FC 层
    tmlc_fully_connected_s8(lstm_sequence, fc_weights, fc_bias, output,
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
