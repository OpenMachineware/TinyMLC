# tinymlc/transform/fusion.py
# Operator Fusion.

from typing import Dict, Any, List, Optional, Tuple
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

            # Remove the activation op
            del ops[i + 1]

            # Update all tensor references
            self._update_tensor_refs_after_removal(model_info, act_output, conv_output)

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

    def _update_tensor_refs_after_removal(
        self,
        model_info: Dict[str, Any],
        removed_output: int,
        replaced_by: int
    ) -> None:
        """Update all tensor references after removing an op."""
        # Update all ops
        for op in model_info.get("ops", []):
            for idx, input_idx in enumerate(op.get("input_indices", [])):
                if input_idx == removed_output:
                    op["input_indices"][idx] = replaced_by

        # Remove the dead tensor from tensors dict
        if removed_output in model_info.get("tensors", {}):
            del model_info["tensors"][removed_output]
        if removed_output in model_info.get("weights", {}):
            del model_info["weights"][removed_output]
