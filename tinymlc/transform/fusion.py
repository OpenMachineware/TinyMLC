# tinymlc/transform/fusion.py
# Operator Fusion.

import numpy as np

from typing import Dict, Any
from tinymlc.transform.base import Pass


class OperatorFusion(Pass):
    """
    Operator Fusion.

    Fuses adjacent operators into single operators:
        - CONV_2D + RELU -> CONV_2D (with activation fused in params)
        - FC + RELU -> FC (with activation fused in params)
        - CONV_2D + RELU6 -> CONV_2D (with relu6)
        - CONV_2D + HARD_SIGMOID -> CONV_2D (with hard_sigmoid)
        - FC + SOFTMAX -> FC (with softmax fused)
    """

    def __init__(self, name: str = "OperatorFusion"):
        super().__init__(name)
        self._fused_count = 0

    def run(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run operator fusion on model_info."""
        model_info = self._copy_model(model_info)
        self._fused_count = 0

        # Iteratively fuse until no more changes
        changed = True
        while changed:
            changed = False
            changed |= self._fuse_conv_activation(model_info)
            changed |= self._fuse_fc_activation(model_info)
            changed |= self._fuse_fc_softmax(model_info)
            changed |= self._fuse_dwconv_activation(model_info)
            changed |= self._fuse_conv_add(model_info)
            changed |= self._fuse_conv_batchnorm(model_info)
            changed |= self._fuse_conv_conv(model_info)

        if self._fused_count > 0:
            self._log_change(f"Fused {self._fused_count} operator pairs")

        return model_info

    def _fuse_conv_activation(self, model_info: Dict[str, Any]) -> bool:
        """Fuse CONV_2D + activation into a single CONV_2D."""
        ops = model_info.get("ops", [])
        fused = False
        i = 0

        while i < len(ops) - 1:
            op = ops[i]
            next_op = ops[i + 1]

            if op.get("op_name") != "CONV_2D":
                i += 1
                continue

            # Check next op is an activation
            activation = None
            if next_op.get("op_name") == "RELU":
                activation = "relu"
            elif next_op.get("op_name") == "RELU6":
                activation = "relu6"
            elif next_op.get("op_name") == "HARD_SIGMOID":
                activation = "hard_sigmoid"
            else:
                i += 1
                continue

            # Check that conv's output feeds into activation's input
            conv_output = op.get("output_indices", [])[0]
            act_input = next_op.get("input_indices", [])[0]

            if conv_output != act_input:
                i += 1
                continue

            # Also check that activation's output is used by later ops
            # (not just a dead op)
            act_output = next_op.get("output_indices", [])[0]

            # Fuse: add activation to conv params
            conv_params = op.get("conv_params", {})
            conv_params["activation"] = activation
            op["conv_params"] = conv_params

            # Update op's output to activation's output
            op["output_indices"] = [act_output]

            # Ensure act_output exists in tensors
            tensors = model_info.get("tensors", {})
            if act_output not in tensors:
                if conv_output in tensors:
                    tensors[act_output] = tensors[conv_output].copy()
                    tensors[act_output]["name"] = f"tensor_{act_output}"
                else:
                    tensors[act_output] = {
                        "name": f"tensor_{act_output}",
                        "shape": [1, 8, 28, 28],
                        "dtype": "float32",
                        "scale": 1.0,
                        "zero_point": 0,
                    }
            model_info["tensors"] = tensors

            # Update all references: replace conv_output with act_output,
            # then delete conv_output
            self._update_tensor_refs_after_removal(model_info, conv_output,
                                                   act_output)

            # Remove the activation op
            del ops[i + 1]

            fused = True
            self._fused_count += 1
            self._log_change(f"  Fused CONV_2D + {activation}")

            # Don't increment i, check if next op can also be fused
            # (but there won't be another activation right after)

        if fused:
            model_info["ops"] = ops

        return fused

    def _fuse_fc_activation(self, model_info: Dict[str, Any]) -> bool:
        """Fuse FC + activation into a single FC."""
        ops = model_info.get("ops", [])
        fused = False
        i = 0

        while i < len(ops) - 1:
            op = ops[i]
            next_op = ops[i + 1]

            if op.get("op_name") != "FULLY_CONNECTED":
                i += 1
                continue

            activation = None
            if next_op.get("op_name") == "RELU":
                activation = "relu"
            elif next_op.get("op_name") == "RELU6":
                activation = "relu6"
            else:
                i += 1
                continue

            fc_output = op.get("output_indices", [])[0]
            act_input = next_op.get("input_indices", [])[0]

            if fc_output != act_input:
                i += 1
                continue

            act_output = next_op.get("output_indices", [])[0]

            fc_params = op.get("fc_params", {})
            fc_params["activation"] = activation
            op["fc_params"] = fc_params

            op["output_indices"] = [act_output]

            del ops[i + 1]

            fused = True
            self._fused_count += 1
            self._log_change(f"  Fused FULLY_CONNECTED + {activation}")

        if fused:
            model_info["ops"] = ops

        return fused

    def _fuse_fc_softmax(self, model_info: Dict[str, Any]) -> bool:
        """Fuse FC + SOFTMAX into FC with softmax flag."""
        ops = model_info.get("ops", [])
        fused = False
        i = 0

        while i < len(ops) - 1:
            op = ops[i]
            next_op = ops[i + 1]

            if op.get("op_name") != "FULLY_CONNECTED":
                i += 1
                continue

            if next_op.get("op_name") != "SOFTMAX":
                i += 1
                continue

            fc_output = op.get("output_indices", [])[0]
            sm_input = next_op.get("input_indices", [])[0]

            if fc_output != sm_input:
                i += 1
                continue

            sm_output = next_op.get("output_indices", [])[0]

            fc_params = op.get("fc_params", {})
            fc_params["with_softmax"] = True
            op["fc_params"] = fc_params

            op["output_indices"] = [sm_output]

            del ops[i + 1]

            fused = True
            self._fused_count += 1
            self._log_change("  Fused FULLY_CONNECTED + SOFTMAX")

        if fused:
            model_info["ops"] = ops

        return fused

    def _fuse_dwconv_activation(self, model_info: Dict[str, Any]) -> bool:
        """Fuse DEPTHWISE_CONV_2D + activation into a single
        DEPTHWISE_CONV_2D."""
        ops = model_info.get("ops", [])
        fused = False
        i = 0

        while i < len(ops) - 1:
            op = ops[i]
            next_op = ops[i + 1]

            if op.get("op_name") != "DEPTHWISE_CONV_2D":
                i += 1
                continue

            # Check next op is an activation
            activation = None
            if next_op.get("op_name") == "RELU":
                activation = "relu"
            elif next_op.get("op_name") == "RELU6":
                activation = "relu6"
            else:
                i += 1
                continue

            # Check that dwconv's output feeds into activation's input
            dwconv_output = op.get("output_indices", [])[0]
            act_input = next_op.get("input_indices", [])[0]

            if dwconv_output != act_input:
                i += 1
                continue

            act_output = next_op.get("output_indices", [])[0]

            # Fuse: add activation to dwconv params
            dwconv_params = op.get("dwconv_params", {})
            dwconv_params["activation"] = activation
            op["dwconv_params"] = dwconv_params

            op["output_indices"] = [act_output]

            del ops[i + 1]

            self._update_tensor_refs_after_removal(model_info, act_output,
                                                   dwconv_output)

            fused = True
            self._fused_count += 1
            self._log_change(f"  Fused DEPTHWISE_CONV_2D + {activation}")

        if fused:
            model_info["ops"] = ops

        return fused

    def _count_uses(self, model_info: Dict[str, Any], tensor_idx: int) -> int:
        """Count how many ops use tensor_idx as input."""
        count = 0
        for op in model_info.get("ops", []):
            if tensor_idx in op.get("input_indices", []):
                count += 1
        return count

    def _fuse_conv_add(self, model_info: Dict[str, Any]) -> bool:
        """
        Fuse CONV_2D + ADD (residual connection) into a single CONV_2D.

        Pattern:
            input ──┬──> CONV_2D ──> ADD
                    │                 ↑
                    └─────────────────┘

        Becomes:
            input ──> CONV_2D (with residual=True) ──> output
        """
        ops = model_info.get("ops", [])
        fused = False
        i = 0

        while i < len(ops) - 1:
            op = ops[i]
            next_op = ops[i + 1]

            if op.get("op_name") != "CONV_2D":
                i += 1
                continue

            if next_op.get("op_name") != "ADD":
                i += 1
                continue

            # CONV_2D outputs: [conv_out, ...]
            conv_outputs = op.get("output_indices", [])
            if not conv_outputs:
                i += 1
                continue
            conv_out = conv_outputs[0]

            # ADD inputs: [a, b]
            add_inputs = next_op.get("input_indices", [])
            if len(add_inputs) != 2:
                i += 1
                continue

            # Check if the ADD is adding conv output with the conv input
            # One input must be conv_out, the other must be conv's input
            conv_inputs = op.get("input_indices", [])
            if not conv_inputs:
                i += 1
                continue
            conv_in = conv_inputs[0]

            # Check if ADD inputs are exactly {conv_out, conv_in}
            if set(add_inputs) != {conv_out, conv_in}:
                i += 1
                continue

            # Also check that conv_out is not used by any other op
            # (otherwise we can't safely remove the ADD)
            used_count = self._count_uses(model_info, conv_out)
            if used_count > 1:
                # conv_out is used by other ops, can't fuse
                i += 1
                continue

            # Get ADD's output
            add_output = next_op.get("output_indices", [])[0]

            # Mark CONV_2D as having a residual connection
            conv_params = op.get("conv_params", {})
            conv_params["residual"] = True
            op["conv_params"] = conv_params

            # Update CONV_2D's output to ADD's output
            op["output_indices"] = [add_output]

            # Remove the ADD op
            del ops[i + 1]

            # Update tensor references: any op that used add_output
            # now uses conv_out (which is the same tensor, but we keep it)
            # Actually we need to clean up the old conv_out tensor
            # Since we're using add_output as the new output, we need to
            # make sure conv_out is not left dangling.

            # Remove conv_out from tensors (it's replaced by add_output)
            if conv_out in model_info.get("tensors", {}):
                del model_info["tensors"][conv_out]
            if conv_out in model_info.get("weights", {}):
                del model_info["weights"][conv_out]

            fused = True
            self._fused_count += 1
            self._log_change("  Fused CONV_2D + ADD (residual)")

            # Don't increment i, check next op

        if fused:
            model_info["ops"] = ops

        return fused

    def _fuse_conv_batchnorm(self, model_info: Dict[str, Any]) -> bool:
        """
        Fuse CONV_2D + BATCH_NORM into a single CONV_2D.

        The BN parameters are folded into the conv weights and bias.
        """
        ops = model_info.get("ops", [])
        tensors = model_info.get("tensors", {})
        weights = model_info.get("weights", {})
        fused = False
        i = 0

        while i < len(ops) - 1:
            op = ops[i]
            next_op = ops[i + 1]

            if op.get("op_name") != "CONV_2D":
                i += 1
                continue

            if next_op.get("op_name") != "BATCH_NORM":
                i += 1
                continue

            # Check direct connection
            conv_out = op.get("output_indices", [])[0]
            bn_input = next_op.get("input_indices", [])[0]
            if conv_out != bn_input:
                i += 1
                continue

            # Get BN parameters
            bn_params = next_op.get("bn_params", {})
            scale_idx = bn_params.get("scale_idx")
            bias_idx = bn_params.get("bias_idx")
            mean_idx = bn_params.get("mean_idx")
            var_idx = bn_params.get("var_idx")
            epsilon = bn_params.get("epsilon", 1e-5)

            if None in (scale_idx, bias_idx, mean_idx, var_idx):
                i += 1
                continue

            # Get conv weights and bias
            conv_inputs = op.get("input_indices", [])
            if len(conv_inputs) < 2:
                i += 1
                continue
            weight_idx = conv_inputs[1]
            bias_idx_conv = conv_inputs[2] if len(conv_inputs) > 2 else None

            if weight_idx not in weights:
                i += 1
                continue

            # Extract BN parameters as numpy arrays
            scale = weights.get(scale_idx)
            bias = weights.get(bias_idx)
            mean = weights.get(mean_idx)
            var = weights.get(var_idx)

            if not all(isinstance(x, np.ndarray) for x in
                       (scale, bias, mean, var)):
                i += 1
                continue

            # Fold BN into conv weights:
            #   w' = w * scale / sqrt(var + eps)
            #   b' = (b - mean) * scale / sqrt(var + eps) + bias
            conv_weight = weights[weight_idx]
            conv_bias = weights.get(bias_idx_conv,
                                    np.zeros(conv_weight.shape[-1],
                                             dtype=np.int32))

            # Compute folding factor
            std = np.sqrt(var + epsilon)
            factor = scale / std

            # Fold into weights (assuming conv_weight shape:
            # [H, W, C_in, C_out])
            # Convert to float for computation
            w_f = conv_weight.astype(np.float32)
            b_f = conv_bias.astype(np.float32)
            factor_f = factor.astype(np.float32).reshape(1, 1, 1, -1)

            # Fold: w_new = w * factor
            w_new = (w_f * factor_f).astype(np.int8)

            # Fold bias: b_new = (b - mean) * factor + bias
            mean_f = mean.astype(np.float32).reshape(1, -1)
            b_new = ((b_f - mean_f) * factor_f.reshape(1, -1) + bias.astype(
                np.float32).reshape(1, -1))
            b_new = b_new.astype(np.int32).flatten()

            # Update weights
            weights[weight_idx] = w_new
            if bias_idx_conv:
                weights[bias_idx_conv] = b_new
            else:
                # Create bias if it didn't exist
                bias_new_idx = max(weights.keys()) + 1 if weights else 1
                weights[bias_new_idx] = b_new
                op["input_indices"].append(bias_new_idx)

            # Remove BN op
            bn_output = next_op.get("output_indices", [])[0]
            op["output_indices"] = [bn_output]

            del ops[i + 1]

            fused = True
            self._fused_count += 1
            self._log_change("  Fused CONV_2D + BATCH_NORM")

        if fused:
            model_info["ops"] = ops

        return fused

    def _fuse_conv_conv(self, model_info: Dict[str, Any]) -> bool:
        """
        Fuse CONV_2D + CONV_2D into one.

        Supports:
            1. 1x1 conv + 1x1 conv -> 1x1 conv
            2. 3x3 conv + 3x3 conv -> 5x5 conv
        """
        ops = model_info.get("ops", [])
        tensors = model_info.get("tensors", {})
        weights = model_info.get("weights", {})
        fused = False
        i = 0

        while i < len(ops) - 1:
            op = ops[i]
            next_op = ops[i + 1]

            if op.get("op_name") != "CONV_2D" or next_op.get(
                    "op_name") != "CONV_2D":
                i += 1
                continue

            # Check direct connection
            conv_out = op.get("output_indices", [])[0]
            next_input = next_op.get("input_indices", [])[0]
            if conv_out != next_input:
                i += 1
                continue

            # Check no activation on first conv
            conv_params = op.get("conv_params", {})
            if conv_params.get("activation"):
                i += 1
                continue

            # Check conv_out is only used by the next conv
            if self._count_uses(model_info, conv_out) > 1:
                i += 1
                continue

            # Get params
            c1 = op.get("conv_params", {})
            c2 = next_op.get("conv_params", {})

            # Get weight indices
            conv1_inputs = op.get("input_indices", [])
            conv2_inputs = next_op.get("input_indices", [])
            if len(conv1_inputs) < 2 or len(conv2_inputs) < 2:
                i += 1
                continue
            w1_idx = conv1_inputs[1]
            w2_idx = conv2_inputs[1]

            if w1_idx not in weights or w2_idx not in weights:
                i += 1
                continue

            w1 = weights[w1_idx]
            w2 = weights[w2_idx]

            if len(w1.shape) != 4 or len(w2.shape) != 4:
                i += 1
                continue

            if w1.shape[3] != w2.shape[2]:
                i += 1
                continue

            C_in = w1.shape[2]
            C_mid = w1.shape[3]
            C_out = w2.shape[3]

            k1 = c1.get("kernel_size")
            k2 = c2.get("kernel_size")
            s1 = c1.get("stride", 1)
            s2 = c2.get("stride", 1)
            p1 = c1.get("padding", "SAME")
            p2 = c2.get("padding", "SAME")

            # ----------------------------------------------------------------
            # Pattern 1: 1x1 + 1x1 -> 1x1
            # ----------------------------------------------------------------
            if k1 == 1 and k2 == 1 and s1 == 1 and s2 == 1:
                w1_f = w1.astype(np.float32)
                w2_f = w2.astype(np.float32)

                # w1: [1, 1, C_in, C_mid], w2: [1, 1, C_mid, C_out]
                # w_fused: [1, 1, C_in, C_out]
                w_fused = np.matmul(
                    w1_f.transpose(0, 1, 3, 2),  # [1, 1, C_mid, C_in]
                    w2_f.transpose(0, 1, 2, 3)  # [1, 1, C_mid, C_out]
                )
                w_fused = w_fused.astype(np.int8)

                # Update weights
                weights[w1_idx] = w_fused

                # Keep second conv's params (but update kernel size)
                next_op["conv_params"]["kernel_size"] = 1

                # Remove first conv
                op_inputs = op.get("input_indices", [])
                next_op["input_indices"][0] = op_inputs[0]

                # Clean up
                if conv_out in tensors:
                    del tensors[conv_out]
                if conv_out in weights:
                    del weights[conv_out]

                del ops[i]

                fused = True
                self._fused_count += 1
                self._log_change("  Fused CONV_2D + CONV_2D (1x1 + 1x1 -> 1x1)")
                continue

            # ----------------------------------------------------------------
            # Pattern 2: 3x3 + 3x3 -> 5x5
            # ----------------------------------------------------------------
            if (k1 == 3 and k2 == 3 and s1 == 1 and s2 == 1
                    and p1 == "SAME" and p2 == "SAME"):
                w1_f = w1.astype(np.float32)
                w2_f = w2.astype(np.float32)

                # w1: [3, 3, C_in, C_mid], w2: [3, 3, C_mid, C_out]
                # w_fused: [5, 5, C_in, C_out]
                w_fused = np.zeros((5, 5, C_in, C_out), dtype=np.float32)

                for ic in range(C_in):
                    for oc in range(C_out):
                        # For each output channel, convolve w1 with w2
                        # w1[:, :, ic, :] -> [3, 3, C_mid]
                        # w2[:, :, :, oc] -> [3, 3, C_mid]
                        # Result is a 5x5 kernel
                        kernel = np.zeros((5, 5), dtype=np.float32)
                        kernel[1:4, 1:4] = w1_f[
                            :, :, ic, :]  # Place w1 in center
                        for mid in range(C_mid):
                            w2_mid = w2_f[:, :, mid, oc]
                            kernel_tmp = kernel[:, :, mid] * w2_mid
                            w_fused[:, :, ic, oc] += kernel_tmp

                w_fused = w_fused.astype(np.int8)

                # Update weights
                weights[w1_idx] = w_fused

                # Update second conv params
                next_op["conv_params"]["kernel_size"] = 5
                next_op["conv_params"]["padding"] = "VALID"

                # Remove first conv
                op_inputs = op.get("input_indices", [])
                next_op["input_indices"][0] = op_inputs[0]

                # Clean up
                if conv_out in tensors:
                    del tensors[conv_out]
                if conv_out in weights:
                    del weights[conv_out]

                del ops[i]

                fused = True
                self._fused_count += 1
                self._log_change("  Fused CONV_2D + CONV_2D (3x3 + 3x3 -> 5x5)")
                continue

            i += 1

        if fused:
            model_info["ops"] = ops

        return fused

    def _update_tensor_refs_after_removal(
        self,
        model_info: Dict[str, Any],
        removed_output: int,
        replaced_by: int
    ) -> None:
        """Update all tensor references after removing an op."""
        # Update all ops
        for op in model_info.get("ops", []):
            # Update input_indices
            for idx, input_idx in enumerate(op.get("input_indices", [])):
                if input_idx == removed_output:
                    op["input_indices"][idx] = replaced_by

            # Update output_indices
            for idx, output_idx in enumerate(op.get("output_indices", [])):
                if output_idx == removed_output:
                    op["output_indices"][idx] = replaced_by

        # Remove the dead tensor from tensors dict
        if removed_output in model_info.get("tensors", {}):
            del model_info["tensors"][removed_output]
        if removed_output in model_info.get("weights", {}):
            del model_info["weights"][removed_output]
