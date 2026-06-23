#include "tinymlc.h"
#include "lut.h"
#include "debug_print.h"

// Default macros
#ifndef LSTM_SHIFT_I
#define LSTM_SHIFT_I 8
#endif
#ifndef LSTM_SHIFT_F
#define LSTM_SHIFT_F 8
#endif
#ifndef LSTM_SHIFT_G
#define LSTM_SHIFT_G 8
#endif
#ifndef LSTM_SHIFT_O
#define LSTM_SHIFT_O 8
#endif


void tmlc_unidirectional_sequence_lstm_s8(
    const int8_t* input,
    const int8_t* input_weights,
    const int8_t* recurrent_weights,
    const int32_t* bias,
    int8_t* output_sequence,   // If NULL, don't save full sequence
    int8_t* output_state,      // If NULL, don't save final state
    int8_t* cell_state,        // If NULL, don't save cell state
    int time_steps,
    int batch_size,
    int input_size,
    int hidden_size)
{
    // ========== 1. Internal State ==========
    int8_t h_cur[hidden_size];
    int8_t c_cur[hidden_size];

    // Initialize state
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

    // ========== 2. Temporary Buffer ==========
    int32_t gate_i[hidden_size];
    int32_t gate_f[hidden_size];
    int32_t gate_g[hidden_size];
    int32_t gate_o[hidden_size];

    int16_t act_i[hidden_size];
    int16_t act_f[hidden_size];
    int16_t act_g[hidden_size];
    int16_t act_o[hidden_size];

    // ========== 3. Weight Pointer ==========
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

    // ========== 4. Time Step Loop ==========
    for (int t = 0; t < time_steps; t++) {
        // Input gate (i)
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

        // Forget gate (f)
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

        // Candidate memory gate (g)
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

        // Output gate (o)
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

        // Activation function
        for (int i = 0; i < hidden_size; i++) {
            act_i[i] = sigmoid_lut_lookup(gate_i[i] >> LSTM_SHIFT_I);
            act_f[i] = sigmoid_lut_lookup(gate_f[i] >> LSTM_SHIFT_F);
            act_g[i] = tanh_lut_lookup(gate_g[i] >> LSTM_SHIFT_G);
            act_o[i] = sigmoid_lut_lookup(gate_o[i] >> LSTM_SHIFT_O);
        }

        // Update state
        // TODO: The >> 8 here is hardcoded, should use actual quantization scale
        // cell state update: c = f * c_prev + i * g
        // hidden state update: h = o * tanh(c)
        // Need to calculate multiplier and shift based on each state's scale
        for (int i = 0; i < hidden_size; i++) {
            int32_t new_c = ((int32_t)act_f[i] * (int32_t)c_cur[i]) >> 8;
            new_c += ((int32_t)act_i[i] * (int32_t)act_g[i]) >> 8;
            c_cur[i] = (int8_t)(new_c >> 8);

            int32_t tanh_c = tanh_lut_lookup((int32_t)c_cur[i] * 32);
            int32_t new_h = ((int32_t)act_o[i] * tanh_c) >> 8;
            h_cur[i] = (int8_t)(new_h >> 8);
        }

        // Save full sequence
        if (output_sequence) {
            int8_t* seq_out = output_sequence + t * hidden_size;
            for (int i = 0; i < hidden_size; i++) {
                seq_out[i] = h_cur[i];
            }
        }

        x_ptr += input_size;
    }

    // Write back final state
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
