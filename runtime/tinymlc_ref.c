#include "tinymlc.h"
#include "lut.h"
#include "model.h"

void tmlc_fully_connected_s8(const int8_t* input,
                             const int8_t* weights,
                             const int32_t* bias,
                             int8_t* output,
                             int input_size,
                             int output_size) {
    for (int out = 0; out < output_size; out++) {
        int32_t sum = bias ? bias[out] : 0;
        for (int in = 0; in < input_size; in++) {
            sum += (int32_t)input[in] * (int32_t)weights[out * input_size + in];
        }
        // FIXME 简单的量化缩放（后续可以优化）
        output[out] = (int8_t)(sum >> 8);
    }
}

void tmlc_softmax_s8(const int8_t* input, int8_t* output, int size) {
    // 找最大值
    int8_t max = -128;
    for (int i = 0; i < size; i++) {
        if (input[i] > max) max = input[i];
    }

    // 计算 exp 近似值（整数版本）
    int32_t sum = 0;
    int32_t exp_vals[size];
    for (int i = 0; i < size; i++) {
        int32_t x = (int32_t)(input[i] - max);
        int32_t exp_val = (x >= 0) ? (1 << (x / 4)) : (1 >> ((-x) / 4));
        exp_vals[i] = exp_val;
        sum += exp_val;
    }

    // 归一化
    for (int i = 0; i < size; i++) {
        int32_t prob = (exp_vals[i] * 128) / sum;
        output[i] = (int8_t)prob;
    }
}

// LSTM 量化参数（从模型元数据提取，这里使用典型值）
// 实际应该从 tflite 的 quantization 字段读取
#define LSTM_INPUT_SCALE   0.00390625f   // 1/256
#define LSTM_INPUT_ZP      0
#define LSTM_STATE_SCALE   0.00390625f
#define LSTM_STATE_ZP      0

void tmlc_unidirectional_sequence_lstm_s8(
    const int8_t* input,
    const int8_t* input_weights,
    const int8_t* recurrent_weights,
    const int32_t* bias,
    int8_t* output_sequence,   // 如果为 NULL，则不保存完整序列
    int8_t* output_state,      // 如果为 NULL，则不保存最后状态
    int8_t* cell_state,        // 如果为 NULL，则不保存细胞状态
    int time_steps,
    int batch_size,
    int input_size,
    int hidden_size)
{
    // ========== 1. 内部状态 ==========
    int8_t h_cur[hidden_size];
    int8_t c_cur[hidden_size];

    // 初始化状态
    if (output_state) {
        for (int i = 0; i < hidden_size; i++) {
            h_cur[i] = output_state[i];
        }
    } else {
        for (int i = 0; i < hidden_size; i++) {
            h_cur[i] = 0;
        }
    }

    if (cell_state) {
        for (int i = 0; i < hidden_size; i++) {
            c_cur[i] = cell_state[i];
        }
    } else {
        for (int i = 0; i < hidden_size; i++) {
            c_cur[i] = 0;
        }
    }

    // ========== 2. 临时缓冲区 ==========
    int32_t gate_i[hidden_size];
    int32_t gate_f[hidden_size];
    int32_t gate_g[hidden_size];
    int32_t gate_o[hidden_size];

    int16_t act_i[hidden_size];
    int16_t act_f[hidden_size];
    int16_t act_g[hidden_size];
    int16_t act_o[hidden_size];

    // ========== 3. 权重指针 ==========
    const int8_t* wi_ptr = input_weights;
    const int8_t* wf_ptr = input_weights + hidden_size * input_size;
    const int8_t* wg_ptr = input_weights + 2 * hidden_size * input_size;
    const int8_t* wo_ptr = input_weights + 3 * hidden_size * input_size;

    const int8_t* ri_ptr = recurrent_weights;
    const int8_t* rf_ptr = recurrent_weights + hidden_size * hidden_size;
    const int8_t* rg_ptr = recurrent_weights + 2 * hidden_size * hidden_size;
    const int8_t* ro_ptr = recurrent_weights + 3 * hidden_size * hidden_size;

    const int32_t* bi_ptr = bias;
    const int32_t* bf_ptr = bias + hidden_size;
    const int32_t* bg_ptr = bias + 2 * hidden_size;
    const int32_t* bo_ptr = bias + 3 * hidden_size;

    const int8_t* x_ptr = input;

    // ========== 4. 时间步循环 ==========
    for (int t = 0; t < time_steps; t++) {
        // 输入门 (i)
        for (int i = 0; i < hidden_size; i++) {
            int32_t sum = bi_ptr[i];
            for (int j = 0; j < input_size; j++) {
                sum += (int32_t)x_ptr[j] * (int32_t)wi_ptr[i * input_size + j];
            }
            for (int j = 0; j < hidden_size; j++) {
                sum += (int32_t)h_cur[j] * (int32_t)ri_ptr[i * hidden_size + j];
            }
            gate_i[i] = sum;
        }

        // 遗忘门 (f)
        for (int i = 0; i < hidden_size; i++) {
            int32_t sum = bf_ptr[i];
            for (int j = 0; j < input_size; j++) {
                sum += (int32_t)x_ptr[j] * (int32_t)wf_ptr[i * input_size + j];
            }
            for (int j = 0; j < hidden_size; j++) {
                sum += (int32_t)h_cur[j] * (int32_t)rf_ptr[i * hidden_size + j];
            }
            gate_f[i] = sum;
        }

        // 候选记忆门 (g)
        for (int i = 0; i < hidden_size; i++) {
            int32_t sum = bg_ptr[i];
            for (int j = 0; j < input_size; j++) {
                sum += (int32_t)x_ptr[j] * (int32_t)wg_ptr[i * input_size + j];
            }
            for (int j = 0; j < hidden_size; j++) {
                sum += (int32_t)h_cur[j] * (int32_t)rg_ptr[i * hidden_size + j];
            }
            gate_g[i] = sum;
        }

        // 输出门 (o)
        for (int i = 0; i < hidden_size; i++) {
            int32_t sum = bo_ptr[i];
            for (int j = 0; j < input_size; j++) {
                sum += (int32_t)x_ptr[j] * (int32_t)wo_ptr[i * input_size + j];
            }
            for (int j = 0; j < hidden_size; j++) {
                sum += (int32_t)h_cur[j] * (int32_t)ro_ptr[i * hidden_size + j];
            }
            gate_o[i] = sum;
        }

        // 激活函数
        for (int i = 0; i < hidden_size; i++) {
            act_i[i] = sigmoid_lut_lookup(gate_i[i] >> LSTM_SHIFT_I);
            act_f[i] = sigmoid_lut_lookup(gate_f[i] >> LSTM_SHIFT_F);
            act_g[i] = tanh_lut_lookup(gate_g[i] >> LSTM_SHIFT_G);
            act_o[i] = sigmoid_lut_lookup(gate_o[i] >> LSTM_SHIFT_O);
        }

        // 更新状态
        for (int i = 0; i < hidden_size; i++) {
            int32_t new_c = ((int32_t)act_f[i] * (int32_t)c_cur[i]) >> 8;
            new_c += ((int32_t)act_i[i] * (int32_t)act_g[i]) >> 8;
            c_cur[i] = (int8_t)(new_c >> 8);

            int32_t tanh_c = tanh_lut_lookup((int32_t)c_cur[i] * 32);
            int32_t new_h = ((int32_t)act_o[i] * tanh_c) >> 8;
            h_cur[i] = (int8_t)(new_h >> 8);
        }

        // 保存完整序列
        if (output_sequence) {
            int8_t* seq_out = output_sequence + t * hidden_size;
            for (int i = 0; i < hidden_size; i++) {
                seq_out[i] = h_cur[i];
            }
        }

        x_ptr += input_size;
    }

    // 写回最终状态
    if (output_state) {
        for (int i = 0; i < hidden_size; i++) {
            output_state[i] = h_cur[i];
        }
    }
    if (cell_state) {
        for (int i = 0; i < hidden_size; i++) {
            cell_state[i] = c_cur[i];
        }
    }
}

void tmlc_reshape_s8(const int8_t* input, int8_t* output,
                     int input_size, const int* new_shape, int shape_size)
{
    // 计算输出大小
    int output_size = 1;
    for (int i = 0; i < shape_size; i++) {
        output_size *= new_shape[i];
    }

    // 验证大小匹配
    if (input_size != output_size) {
        // 大小不匹配，报错或直接返回
        // 在 MCU 上可以触发一个错误标志
        return;
    }

    // 复制数据
    for (int i = 0; i < input_size; i++) {
        output[i] = input[i];
    }
    // 或用 memcpy
    // memcpy(output, input, input_size * sizeof(int8_t));
}
