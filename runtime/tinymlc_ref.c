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
    const int8_t* input,           // [time_steps, batch, input_size]
    const int8_t* input_weights,
    const int8_t* recurrent_weights,
    const int32_t* bias,
    int8_t* output_state,          // [batch, hidden_size] 输出状态（隐藏状态）
    int8_t* cell_state,            // [batch, hidden_size] 细胞状态
    int time_steps,
    int batch_size,
    int input_size,
    int hidden_size)
{
    // 注意：权重和偏置使用 lstm_weights.h 中的全局变量
    // 注意：input_weights, recurrent_weights, bias 是四个门拼接在一起的
    // 顺序是：i, f, g, o
    // 每个门的权重大小：input_weights: [hidden, input_size], 共 4 * hidden * input_size
    //           recurrent_weights: [hidden, hidden], 共 4 * hidden * hidden
    //           bias: [4 * hidden]

    // 临时缓冲区：当前时间步的门预激活值
    int32_t gate_i[hidden_size];
    int32_t gate_f[hidden_size];
    int32_t gate_g[hidden_size];
    int32_t gate_o[hidden_size];

    // 激活后的门值
    int16_t act_i[hidden_size];
    int16_t act_f[hidden_size];
    int16_t act_g[hidden_size];
    int16_t act_o[hidden_size];

    // 当前时间步的隐藏状态和细胞状态
    int8_t h_cur[hidden_size];
    int8_t c_cur[hidden_size];

    // 复制初始状态
    for (int i = 0; i < hidden_size; i++) {
        h_cur[i] = output_state[i];
        c_cur[i] = cell_state[i];
    }

    const int8_t* x_ptr = input;

    // 权重指针：每个门独立
    // input_weights 布局: [i_weights, f_weights, g_weights, o_weights]
    // 每个都是 [hidden, input_size]
    const int8_t* wi_ptr = input_weights;
    const int8_t* wf_ptr = input_weights + hidden_size * input_size;
    const int8_t* wg_ptr = input_weights + 2 * hidden_size * input_size;
    const int8_t* wo_ptr = input_weights + 3 * hidden_size * input_size;

    // recurrent_weights 布局类似
    const int8_t* ri_ptr = recurrent_weights;
    const int8_t* rf_ptr = recurrent_weights + hidden_size * hidden_size;
    const int8_t* rg_ptr = recurrent_weights + 2 * hidden_size * hidden_size;
    const int8_t* ro_ptr = recurrent_weights + 3 * hidden_size * hidden_size;

    // bias 布局
    const int32_t* bi_ptr = bias;
    const int32_t* bf_ptr = bias + hidden_size;
    const int32_t* bg_ptr = bias + 2 * hidden_size;
    const int32_t* bo_ptr = bias + 3 * hidden_size;

    for (int t = 0; t < time_steps; t++) {
        // ========== 1. 计算四个门的预激活值 ==========

        // 输入门 (i)
        for (int i = 0; i < hidden_size; i++) {
            int32_t sum = bi_ptr[i];
            // 输入权重
            for (int j = 0; j < input_size; j++) {
                sum += (int32_t)x_ptr[j] * (int32_t)wi_ptr[i * input_size + j];
            }
            // 递归权重
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

        // ========== 2. 应用激活函数（LUT） ==========
        for (int i = 0; i < hidden_size; i++) {
            // 输入门：sigmoid
            int32_t x_i = gate_i[i] >> 8;
            if (x_i < -128) x_i = -128;
            if (x_i > 127) x_i = 127;
            act_i[i] = sigmoid_lut_lookup(gate_i[i] >> LSTM_SHIFT_I);

            // 遗忘门：sigmoid
            int32_t x_f = gate_f[i] >> 8;
            if (x_f < -128) x_f = -128;
            if (x_f > 127) x_f = 127;
            act_f[i] = sigmoid_lut_lookup(gate_f[i] >> LSTM_SHIFT_F);

            // 候选记忆：tanh
            int32_t x_g = gate_g[i] >> 8;
            if (x_g < -128) x_g = -128;
            if (x_g > 127) x_g = 127;
            act_g[i] = tanh_lut_lookup(gate_g[i] >> LSTM_SHIFT_G);

            // 输出门：sigmoid
            int32_t x_o = gate_o[i] >> 8;
            if (x_o < -128) x_o = -128;
            if (x_o > 127) x_o = 127;
            act_o[i] = sigmoid_lut_lookup(gate_o[i] >> LSTM_SHIFT_O);
        }

        // ========== 3. 更新细胞状态和隐藏状态 ==========
        for (int i = 0; i < hidden_size; i++) {
            // c[t] = f * c[t-1] + i * g
            int32_t new_c = ((int32_t)act_f[i] * (int32_t)c_cur[i]) >> 8;
            new_c += ((int32_t)act_i[i] * (int32_t)act_g[i]) >> 8;
            c_cur[i] = (int8_t)(new_c >> 8);

            // h[t] = o * tanh(c[t])
            int32_t tanh_c = tanh_lut_lookup((int32_t)c_cur[i] * 32);
            int32_t new_h = ((int32_t)act_o[i] * tanh_c) >> 8;
            h_cur[i] = (int8_t)(new_h >> 8);
        }

        // 移动到下一个时间步
        x_ptr += input_size;
    }

    // 写回最终状态
    for (int i = 0; i < hidden_size; i++) {
        // output_state 是 const，这里强制转换（实际应该是可修改的）
        //((int8_t*)output_state)[i] = h_cur[i];
        output_state[i] = h_cur[i];
        cell_state[i] = c_cur[i];
    }
}
