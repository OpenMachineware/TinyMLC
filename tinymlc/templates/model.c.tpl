// 自动生成的代码，请勿手动修改
// 由 tinymlc 自动生成

#include "tinymlc.h"
{{ includes }}
#include "model.h"
#include "debug_print.h"
#include <stddef.h>

// 输入输出大小
#define INPUT_SIZE {{ input_size }}
#define OUTPUT_SIZE {{ output_size }}

// 中间张量内存（静态分配，放在函数外部）
{% for op in execution_order %}
    {% for out_idx in op.output_indices %}
    int8_t tensor_{{ out_idx }}[{{ tensor_sizes[out_idx] }}] __attribute__((section(".bss")));
    {% endfor %}
    {% if op.op_name == "SVDF" %}
        {% if op.svdf_weights_idx is not none %}
    int8_t tensor_{{ op.svdf_weights_idx }}[{{ tensor_sizes[op.svdf_weights_idx] }}] __attribute__((section(".bss")));
        {% endif %}
        {% if op.svdf_bias_idx is not none %}
    int32_t tensor_{{ op.svdf_bias_idx }}[{{ tensor_sizes[op.svdf_bias_idx] }}] __attribute__((section(".bss")));
        {% endif %}
    {% endif %}
{% endfor %}

{% if has_lstm %}
// LSTM 参数
#define LSTM_TIME_STEPS {{ lstm_time_steps }}
#define LSTM_HIDDEN_SIZE {{ lstm_hidden_size }}
#define TINYMLC_HAS_LSTM
// ... 其他参数
{% endif %}

// 推理函数
void {{ inference_func }}(const int8_t* input, int8_t* output) {
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
        {{ tensor_sizes[op.input_indices[0]] }},
        {{ tensor_sizes[op.output_indices[0]] }}
    );
    {% elif op.op_name == "SOFTMAX" %}
    tmlc_softmax_s8(
        tensor_{{ op.input_indices[0] }},
        tensor_{{ op.output_indices[0] }},
        {{ tensor_sizes[op.input_indices[0]] }}
    );
    {% elif op.op_name == "RESHAPE" %}
    {
        static const int reshape_target[] = { {% for s in op.reshape_target_shape %}{{ s }}{% if not loop.last %}, {% endif %}{% endfor %} };
        int reshape_input_size = {{ tensor_sizes[op.input_indices[0]] }};
        tmlc_reshape_s8(
            tensor_{{ op.input_indices[0] }},
            tensor_{{ op.output_indices[0] }},
            reshape_input_size,
            reshape_target,
            {{ op.reshape_target_shape | length }}
        );
    }
    {% elif op.op_name == "SVDF" %}
    tmlc_svdf_s8(
        tensor_{{ op.data_input_idx }},
        tensor_{{ op.svdf_weights_idx }},
        tensor_{{ op.svdf_bias_idx }},
        tensor_{{ op.output_indices[0] }},
        49,  // time_steps
        257, // input_size
        2,   // rank
        80   // units
    );
    {% endif %}
    {% endfor %}

    // 输出张量映射
    if (output != NULL) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output[i] = tensor_{{ last_output_tensor }}[i];
        }
    }
}
