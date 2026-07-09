# 算子支持状态

## 纯 C 算子（共 31 个）

| 算子名称 | 状态 |
| :--- | :--- |
| conv2d | ✅ 稳定 |
| depthwise_conv2d | ✅ 稳定 |
| max_pool2d | ✅ 稳定 |
| avg_pool2d | ✅ 稳定 |
| global_avg_pool | ✅ 稳定 |
| fc | ✅ 稳定 |
| softmax | ✅ 稳定 |
| relu | ✅ 稳定 |
| leaky_relu | ✅ 稳定 |
| relu6 | ✅ 稳定 |
| prelu | ✅ 稳定 |
| hard_sigmoid | ✅ 稳定 |
| sigmoid | ✅ 稳定 |
| tanh | ✅ 稳定 |
| clip | ✅ 稳定 |
| add | ✅ 稳定 |
| multiply | ✅ 稳定 |
| sub | ✅ 稳定 |
| concat | ✅ 稳定 |
| split | ✅ 稳定 |
| pad | ✅ 稳定 |
| strided_slice | ✅ 稳定 |
| reshape | ✅ 稳定 |
| transpose | ✅ 稳定 |
| flatten | ✅ 稳定 |
| reduce_sum | ✅ 稳定 |
| mean | ✅ 稳定 |
| argmax | ✅ 稳定 |
| nms | ✅ 稳定 |
| upsample | ✅ 稳定 |
| conv_transpose | ✅ 稳定 |

## 加速库支持

### CMSIS-NN（ARM）

| 算子名称 | 状态 |
| :--- | :--- |
| conv2d | ✅ 稳定 |
| depthwise_conv2d | ⚠️ 与纯 C 行为有差异 |
| fc | ✅ 稳定 |
| softmax | ⚠️ 近似实现（查表） |
| max_pool2d | ✅ 稳定 |
| avg_pool2d | ✅ 稳定 |

### NMSIS（RISC-V）

| 算子名称 | 状态 |
| :--- | :--- |
| conv2d | ⚠️ 仅软浮点 |
| depthwise_conv2d | ⚠️ 仅软浮点 |
| fc | ⚠️ 仅软浮点 |
| max_pool2d | ⚠️ 仅软浮点 |
| avg_pool2d | ⚠️ 仅软浮点 |

## 已知问题

- CMSIS-NN depthwise_conv2d 在某些输入尺寸下与纯 C 不一致
- NMSIS 需要硬浮点工具链才能达到最佳性能
- 软浮点 NMSIS 性能提升有限
