// 自动生成的代码，请勿手动修改
// 由 tinymlc 自动生成

#include <string.h>
#include "tinymlc.h"
{{ includes }}

// 输入输出大小
#define INPUT_SIZE {{ input_size }}
#define OUTPUT_SIZE {{ output_size }}

{% if has_lstm %}
// LSTM 参数
#define LSTM_TIME_STEPS {{ lstm_time_steps }}
#define LSTM_HIDDEN_SIZE {{ lstm_hidden_size }}
// ... 其他参数
{% endif %}

// 推理函数
void {{ inference_func }}(const int8_t* input, int8_t* output) {
    // 中间张量内存（简单版：每个输出分配独立数组）
    {% for op in execution_order %}
        {% for out_idx in op.output_indices %}
    static int8_t tensor_{{ out_idx }}[{{ tensor_sizes[out_idx] }}];
        {% endfor %}
    {% endfor %}

    // 输入张量映射
    int8_t* tensor_0 = (int8_t*)input;

    // 按顺序执行算子
    {% for op in execution_order %}
    // {{ op.op_name }} (index: {{ op.index }})
    {% if op.op_name == "UNIDIRECTIONAL_SEQUENCE_LSTM" %}
    tmlc_unidirectional_sequence_lstm_s8(
        tensor_{{ op.input_indices[0] }},
        lstm_input_weights,
        lstm_recurrent_weights,
        lstm_bias,
        tensor_{{ op.output_indices[0] }},
        NULL, NULL,
        {{ op.lstm_params.time_steps }},
        {{ op.lstm_params.batch_size }},
        {{ op.lstm_params.input_size }},
        {{ op.lstm_params.hidden_size }}
    );
    {% elif op.op_name == "FULLY_CONNECTED" %}
    tmlc_fully_connected_s8(
        tensor_{{ op.input_indices[0] }},
        fc_weights,
        fc_bias,
        tensor_{{ op.output_indices[0] }},
        {{ fc_input_size }},
        {{ fc_output_size }}
    );
    {% elif op.op_name == "SOFTMAX" %}
    tmlc_softmax_s8(
        tensor_{{ op.input_indices[0] }},
        tensor_{{ op.output_indices[0] }},
        {{ softmax_size }}
    );
    {% elif op.op_name == "RESHAPE" %}
    tmlc_reshape_s8(
        tensor_{{ op.input_indices[0] }},
        tensor_{{ op.output_indices[0] }},
        {{ reshape_input_size }},
        reshape_target,
        1
    );
    {% endif %}
    {% endfor %}

    // 输出张量映射
    if (output != NULL) {
        memcpy(output, tensor_{{ last_output_tensor }}, OUTPUT_SIZE * sizeof(int8_t));
    }
}
