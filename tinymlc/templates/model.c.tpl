// 自动生成的代码，请勿手动修改
// 由 tinymlc 自动生成

{% if has_lstm %}
#define TINYMLC_HAS_LSTM
#define LSTM_SHIFT_I {{ lstm_shifts[0] }}
#define LSTM_SHIFT_F {{ lstm_shifts[1] }}
#define LSTM_SHIFT_G {{ lstm_shifts[2] }}
#define LSTM_SHIFT_O {{ lstm_shifts[3] }}
{% endif %}

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
    {% elif op.op_name == "ADD" %}
        int8_t tensor_{{ op.add_input1_idx }}[{{ tensor_sizes[op.add_input1_idx] }}] __attribute__((section(".bss")));
        int8_t tensor_{{ op.add_input2_idx }}[{{ tensor_sizes[op.add_input2_idx] }}] __attribute__((section(".bss")));
    {% endif %}
{% endfor %}

{% if has_lstm %}
// LSTM 参数
#define LSTM_TIME_STEPS {{ lstm_time_steps }}
#define LSTM_HIDDEN_SIZE {{ lstm_hidden_size }}
#define TINYMLC_HAS_LSTM
{% endif %}

// 推理函数
{% if inputs_count == 1 %}
void {{ inference_func }}(const int8_t* input, int8_t* output) {
    // 输入张量映射
    int8_t* tensor_0 = (int8_t*)input;
{% elif inputs_count == 2 %}
void {{ inference_func }}(const int8_t* input1, const int8_t* input2, int8_t* output) {
    // 输入张量映射
    int8_t* tensor_0 = (int8_t*)input1;
    int8_t* tensor_1 = (int8_t*)input2;
{% endif %}
    // 按顺序执行算子
    {% for op in execution_order %}
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
        {% elif op.op_name == "ADD" %}
        tmlc_add_s8(
            tensor_{{ op.add_input1_idx }},
            tensor_{{ op.add_input2_idx }},
            tensor_{{ op.output_indices[0] }},
            {{ tensor_sizes[op.output_indices[0]] }}
        );
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
        {% elif op.op_name == "CONV_2D" %}
        tmlc_conv2d_s8(
            tensor_{{ op.data_input_idx }},
            conv_weights,
            conv_bias,
            tensor_{{ op.output_indices[0] }},
            {{ op.conv_params.input_h }},
            {{ op.conv_params.input_w }},
            {{ op.conv_params.input_c }},
            {{ op.conv_params.output_h }},
            {{ op.conv_params.output_w }},
            {{ op.conv_params.output_c }},
            {{ op.conv_params.kernel_h }},
            {{ op.conv_params.kernel_w }},
            {{ op.conv_params.stride_h }},
            {{ op.conv_params.stride_w }},
            0, 0
        );
        {% elif op.op_name == "MAX_POOL_2D" %}
        tmlc_max_pool_2d_s8(
            tensor_{{ op.data_input_idx }},
            tensor_{{ op.output_indices[0] }},
            {{ op.pool_params.input_h }},
            {{ op.pool_params.input_w }},
            {{ op.pool_params.input_c }},
            {{ op.pool_params.output_h }},
            {{ op.pool_params.output_w }},
            {{ op.pool_params.output_c }},
            {{ op.pool_params.pool_size_h }},
            {{ op.pool_params.pool_size_w }},
            {{ op.pool_params.stride_h }},
            {{ op.pool_params.stride_w }},
            0, 0
        );
        {% elif op.op_name == "DEPTHWISE_CONV_2D" %}
        tmlc_depthwise_conv_2d_s8(
            tensor_{{ op.data_input_idx }},
            dw_weights,
            dw_bias,
            tensor_{{ op.output_indices[0] }},
            {{ op.dw_params.input_h }},
            {{ op.dw_params.input_w }},
            {{ op.dw_params.input_c }},
            {{ op.dw_params.output_h }},
            {{ op.dw_params.output_w }},
            {{ op.dw_params.output_c }},
            {{ op.dw_params.kernel_h }},
            {{ op.dw_params.kernel_w }},
            1, 1,
            {{ op.dw_params.depth_multiplier }},
            0, 0
        );
        {% elif op.op_name == "RELU" %}
        tmlc_relu_s8(
            tensor_{{ op.input_indices[0] }},
            tensor_{{ op.output_indices[0] }},
            {{ tensor_sizes[op.input_indices[0]] }}
        );
        {% elif op.op_name == "AVERAGE_POOL_2D" %}
        tmlc_avg_pool_2d_s8(
            tensor_{{ op.data_input_idx }},
            tensor_{{ op.output_indices[0] }},
            {{ op.pool_params.input_h }},
            {{ op.pool_params.input_w }},
            {{ op.pool_params.input_c }},
            {{ op.pool_params.output_h }},
            {{ op.pool_params.output_w }},
            {{ op.pool_params.output_c }},
            {{ op.pool_params.pool_h }},
            {{ op.pool_params.pool_w }},
            {{ op.pool_params.stride_h }},
            {{ op.pool_params.stride_w }},
            0, 0
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
