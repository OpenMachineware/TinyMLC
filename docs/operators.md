# TinyMLC Operators

## Overview

TinyMLC supports multiple operator implementations across three backends:

- **Pure C**: Portable reference implementation
- **CMSIS-NN**: ARM Cortex-M hardware acceleration
- **NMSIS-NN**: RISC-V hardware acceleration

## Status Legend

| Status | Description |
|--------|-------------|
| ✅ | Fully supported, tested |
| ⚠️ | Supported with known quirks (see notes) |
| 🔧 | Needs implementation |
| ❌ | Not supported |

---

## Operator Support Matrix

| Operator | Pure C | CMSIS-NN | NMSIS-NN | Notes |
|----------|--------|----------|----------|-------|
| FULLY_CONNECTED (FC) | ✅ | ✅ | ✅ | |
| CONV_2D | ✅ | ✅ | ✅ | |
| DEPTHWISE_CONV_2D | ✅ | ✅ | ✅ | |
| MAX_POOL_2D | ✅ | ✅ | ✅ | |
| AVERAGE_POOL_2D | ✅ | ✅ | ✅ | |
| GLOBAL_AVERAGE_POOL | ✅ | ✅ | ✅ | |
| SOFTMAX | ✅ | ✅ | ✅ | |
| RELU | ✅ | ✅ | ✅ | |
| RELU6 | ✅ | ✅ | ✅ | |
| LEAKY_RELU | ✅ | ✅ | ✅ | |
| PRELU | ✅ | ✅ | ✅ | |
| HARD_SIGMOID | ✅ | ✅ | ✅ | |
| SIGMOID | ✅ | ✅ | ✅ | |
| TANH | ✅ | ✅ | ✅ | |
| ADD | ✅ | ✅ | ✅ | |
| SUB | ✅ | ✅ | ✅ | |
| MULTIPLY | ✅ | ✅ | ✅ | |
| CONCAT | ✅ | ✅ | ✅ | |
| RESHAPE | ✅ | ✅ | ✅ | |
| TRANSPOSE | ✅ | 🔧 | 🔧 | CMSIS-NN/NMSIS-NN: No hardware support, falls back to Pure C |
| PAD | ✅ | 🔧 | 🔧 | CMSIS-NN/NMSIS-NN: No hardware support, falls back to Pure C |
| MEAN | ✅ | 🔧 | 🔧 | CMSIS-NN/NMSIS-NN: No hardware support, falls back to Pure C |
| REDUCE_SUM | ✅ | 🔧 | 🔧 | CMSIS-NN/NMSIS-NN: No hardware support, falls back to Pure C |
| ARGMAX | ✅ | 🔧 | 🔧 | CMSIS-NN/NMSIS-NN: No hardware support, falls back to Pure C |
| FLATTEN | ✅ | ✅ | ✅ | Simple copy, no acceleration needed |
| SPLIT | ✅ | 🔧 | 🔧 | CMSIS-NN/NMSIS-NN: No hardware support, falls back to Pure C |
| STRIDED_SLICE | ✅ | 🔧 | 🔧 | CMSIS-NN/NMSIS-NN: No hardware support, falls back to Pure C |
| CLIP | ✅ | 🔧 | 🔧 | CMSIS-NN/NMSIS-NN: No hardware support, falls back to Pure C |
| UPSAMPLE | ✅ | 🔧 | 🔧 | CMSIS-NN/NMSIS-NN: No hardware support, falls back to Pure C |
| CONV_TRANSPOSE | ✅ | 🔧 | 🔧 | CMSIS-NN/NMSIS-NN: No hardware support, falls back to Pure C |
| SVDF | ✅ | 🔧 | 🔧 | CMSIS-NN/NMSIS-NN: No hardware support, falls back to Pure C |
| UNIDIRECTIONAL_SEQUENCE_LSTM | ✅ | ✅ | ✅ | |
| NMS | ✅ | 🔧 | 🔧 | CMSIS-NN/NMSIS-NN: No hardware support, falls back to Pure C |

---

## Known Issues

### ⚠️ CMSIS-NN/NMSIS-NN Anomalies

#### 1. RELU6 Function Signature Difference

**Issue**: CMSIS-NN's `arm_relu6_s8` and NMSIS-NN's `riscv_relu6_s8` have different function signatures than expected.

**CMSIS-NN (cmsis_nn/relu6.c)**:
```c
void tmlc_relu6_s8(const int8_t* input, int8_t* output, int size,
                   int32_t zero_point, float input_scale, float output_scale);
```

**Expected (tinymlc.h)**:
```c
void tmlc_relu6_s8(const int8_t* input, int8_t* output, int size,
                   int32_t input_zero_point, int32_t output_zero_point,
                   int32_t input_scale_q, int32_t output_scale_q);
```

**Status**: Implementation includes workaround for size > 16384, using pure C fallback for large tensors.

#### 2. Global Average Pool Scaling

**Issue**: CMSIS-NN's `arm_avgpool_s8` and NMSIS-NN's `riscv_avgpool_s8` use Q15 fixed-point scaling internally.

**Impact**: Output values may differ slightly from pure C implementation due to internal quantization.

**Workaround**: Current implementation uses hardware-accelerated pooling but may have small numerical differences.

---

## Implementation Details

### Directory Structure

```
ops/
├── include/
│   └── tinymlc.h          # Unified operator interface
├── c/                     # Pure C reference implementations
│   ├── fc.c
│   ├── conv2d.c
│   ├── ...
├── arm/
│   ├── cmsis_nn/          # CMSIS-NN accelerated operators
│   │   ├── fc.c
│   │   ├── conv2d.c
│   │   └── ...
│   └── build_arm_*.sh
└── riscv/
    ├── nmsis_nn/          # NMSIS-NN accelerated operators
    │   ├── fc.c
    │   ├── conv2d.c
    │   └── ...
    └── build_riscv_*.sh
```

### Build Configuration

#### ARM Build Options
- `--target arm --accel pure-c`: Pure C implementation
- `--target arm --accel cmsis-nn`: CMSIS-NN accelerated (when available)

#### RISC-V Build Options
- `--target riscv --accel pure-c`: Pure C implementation
- `--target riscv --accel nmsis-nn`: NMSIS-NN accelerated (when available)
- `--target riscv --accel nuclei-ai`: Nuclei AI extension (future)

---

## Adding New Operators

To add a new operator:

1. **Implement Pure C version** in `ops/c/{op_name}.c`
2. **Add function declaration** to `ops/include/tinymlc.h`
3. **Add to codegen** in `tinymlc/codegen.py`
4. **Optional: Add CMSIS-NN version** in `ops/arm/cmsis_nn/{op_name}.c`
5. **Optional: Add NMSIS-NN version** in `ops/riscv/nmsis_nn/{op_name}.c`
6. **Update build scripts** in `ops/*/build_*.sh`
7. **Update this document** with operator status

### Operator Interface Template

```c
// ops/include/tinymlc.h

// int8 version (quantized)
void tmlc_{op_name}_s8(
    const int8_t* input,
    int8_t* output,
    int size,
    // ... other parameters
);

// float32 version (optional)
void tmlc_{op_name}_f32(
    const float* input,
    float* output,
    int size,
    // ... other parameters
);
```
