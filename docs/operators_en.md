Operator Support Status

Pure-C Operators (31 total)

conv2d              ✅ Stable
depthwise_conv2d    ✅ Stable
max_pool2d          ✅ Stable
avg_pool2d          ✅ Stable
global_avg_pool     ✅ Stable
fc                  ✅ Stable
softmax             ✅ Stable
relu                ✅ Stable
leaky_relu          ✅ Stable
relu6               ✅ Stable
prelu               ✅ Stable
hard_sigmoid        ✅ Stable
sigmoid             ✅ Stable
tanh                ✅ Stable
clip                ✅ Stable
add                 ✅ Stable
multiply            ✅ Stable
sub                 ✅ Stable
concat              ✅ Stable
split               ✅ Stable
pad                 ✅ Stable
strided_slice       ✅ Stable
reshape             ✅ Stable
transpose           ✅ Stable
flatten             ✅ Stable
reduce_sum          ✅ Stable
mean                ✅ Stable
argmax              ✅ Stable
nms                 ✅ Stable
upsample            ✅ Stable
conv_transpose      ✅ Stable

Accelerator Support

CMSIS-NN (ARM)

    conv2d              ✅ Stable
    depthwise_conv2d    ⚠️ Behavior differs from pure-C
    fc                  ✅ Stable
    softmax             ⚠️ Approximate (lookup table)
    max_pool2d          ✅ Stable
    avg_pool2d          ✅ Stable

NMSIS (RISC-V)

    conv2d              ⚠️ Soft-float only
    depthwise_conv2d    ⚠️ Soft-float only
    fc                  ⚠️ Soft-float only
    max_pool2d          ⚠️ Soft-float only
    avg_pool2d          ⚠️ Soft-float only

Known Issues

- CMSIS-NN depthwise_conv2d output differs from pure-C on
  certain input sizes
- NMSIS requires hard-float toolchain for best performance
- Soft-float NMSIS provides limited performance improvement
