#include "tinymlc.h"


void tmlc_strided_slice_s8(
    const int8_t* input,
    int8_t* output,
    int in_h, int in_w, int in_c,
    int start_h, int start_w, int start_c,
    int size_h, int size_w, int size_c,
    int stride_h, int stride_w, int stride_c)
{
    for (int oh = 0; oh < size_h; oh++) {
        for (int ow = 0; ow < size_w; ow++) {
            for (int oc = 0; oc < size_c; oc++) {
                int ih = start_h + oh * stride_h;
                int iw = start_w + ow * stride_w;
                int ic = start_c + oc * stride_c;

                if (ih >= 0 && ih < in_h && iw >= 0 &&
                    iw < in_w && ic >= 0 && ic < in_c) {
                    int in_idx = (ih * in_w + iw) * in_c + ic;
                    int out_idx = (oh * size_w + ow) * size_c + oc;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}
