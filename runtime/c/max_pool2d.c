#include "tinymlc.h"

void tmlc_max_pool_2d_s8(const int8_t* input,
                         int8_t* output,
                         int input_h, int input_w, int input_c,
                         int output_h, int output_w, int output_c,
                         int pool_h, int pool_w,
                         int stride_h, int stride_w,
                         int padding_h, int padding_w)
{
    for (int oh = 0; oh < output_h; oh++) {
        for (int ow = 0; ow < output_w; ow++) {
            for (int oc = 0; oc < output_c; oc++) {
                int8_t max_val = -128;
                for (int ph = 0; ph < pool_h; ph++) {
                    for (int pw = 0; pw < pool_w; pw++) {
                        int ih = oh * stride_h + ph - padding_h;
                        int iw = ow * stride_w + pw - padding_w;
                        if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                            int8_t val = input[ih * input_w * input_c + iw * input_c + oc];
                            if (val > max_val) max_val = val;
                        }
                    }
                }
                output[oh * output_w * output_c + ow * output_c + oc] = max_val;
            }
        }
    }
}
