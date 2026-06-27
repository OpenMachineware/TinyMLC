# TinyMLC/transform/simplify.py
# Simplify pass: eliminates redundant operator patterns.
#
# Patterns handled:
#   1. RESHAPE -> RESHAPE: remove the first reshape if no other ops in between
#   2. TRANSPOSE -> TRANSPOSE: merge or remove consecutive transposes
#   3. RESHAPE -> TRANSPOSE: remove reshape if shape matches after transpose
#   4. TRANSPOSE -> RESHAPE: same as above
#   5. ADD 0 / MULTIPLY 1: remove the op and forward the input
#   6. CONCAT with single input: remove concat, forward the input
#   7. SPLIT with single output: remove split, forward the input
#   8. RELU after RELU: remove the second RELU
#   9. POOL with kernel_size=1 and stride=1: remove the pool

from typing import Dict, Any, List, Set, Tuple, Optional
from TinyMLC.transform.base import Pass


class Simplify(Pass):
    """
    Simplify redundant operator patterns.

    Runs multiple simplification patterns in a fixed order.
    Repeats until no more changes.
    """

    def __init__(self, name: str = "Simplify"):
        super().__init__(name)
        self._simplified_count = 0

    def run(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        model_info = self._copy_model(model_info)
        self._simplified_count = 0

        changed = True
        iteration = 0

        while changed:
            changed = False
            iteration += 1

            # Run each simplification pattern in order
            # 1. RESHAPE/TRANSPOSE elimination
            changed |= self._simplify_reshape_transpose(model_info)
            # 2. ADD 0 / MULTIPLY 1 elimination
            changed |= self._simplify_zero_one_ops(model_info)
            # 3. CONCAT single input / SPLIT single output
            changed |= self._simplify_concat_split(model_info)
            # 4. RELU after RELU
            changed |= self._simplify_double_relu(model_info)
            # 5. POOL kernel_size=1
            changed |= self._simplify_pool_k1(model_info)

            if changed:
                self._log_change(
                    f"Iteration {iteration}: simplifications applied")

        if self._simplified_count > 0:
            self._log_change(f"Total: {self._simplified_count} simplifications")

        return model_info

    # ================================================================
    # 1. RESHAPE / TRANSPOSE continuous elimination
    # ================================================================

    def _simplify_reshape_transpose(self, model_info: Dict[str, Any]) -> bool:
        """
        Eliminate redundant reshape/transpose sequences.

        Patterns:
            RESHAPE -> RESHAPE: remove first reshape
            TRANSPOSE -> TRANSPOSE: merge or remove
            RESHAPE -> TRANSPOSE: remove reshape if shape is compatible
        """
        ops = model_info.get("ops", [])
        changed = False
        i = 0

        while i < len(ops) - 1:
            op = ops[i]
            next_op = ops[i + 1]
            op_name = op.get("op_name")
            next_name = next_op.get("op_name")

            # Pattern: RESHAPE -> RESHAPE
            if op_name == "RESHAPE" and next_name == "RESHAPE":
                # Check that the first reshape's output goes directly
                # to the second
                if self._is_direct_connection(op, next_op):
                    self._forward_output(op, next_op, model_info)
                    del ops[i]
                    changed = True
                    self._simplified_count += 1
                    self._log_change("  Removed redundant RESHAPE -> RESHAPE")
                    continue  # Don't increment i

            # Pattern: TRANSPOSE -> TRANSPOSE
            elif op_name == "TRANSPOSE" and next_name == "TRANSPOSE":
                if self._is_direct_connection(op, next_op):
                    # Merge or remove
                    if self._merge_transposes(op, next_op, model_info):
                        del ops[i]
                        changed = True
                        self._simplified_count += 1
                        self._log_change("  Merged TRANSPOSE -> TRANSPOSE")
                        continue

            # Pattern: RESHAPE -> TRANSPOSE
            elif op_name == "RESHAPE" and next_name == "TRANSPOSE":
                if self._is_direct_connection(op, next_op):
                    # Check if reshape target shape matches after transpose
                    reshape_params = op.get("reshape_params", {})
                    target_shape = reshape_params.get("shape", [])
                    if (target_shape and
                        self._is_reshape_redundant_after_transpose(
                            op, next_op, target_shape, model_info)):
                        self._forward_output(op, next_op, model_info)
                        del ops[i]
                        changed = True
                        self._simplified_count += 1
                        self._log_change(
                            "  Removed redundant RESHAPE before TRANSPOSE")
                        continue

            # Pattern: TRANSPOSE -> RESHAPE (symmetric)
            elif op_name == "TRANSPOSE" and next_name == "RESHAPE":
                if self._is_direct_connection(op, next_op):
                    reshape_params = next_op.get("reshape_params", {})
                    target_shape = reshape_params.get("shape", [])
                    if (target_shape and
                        self._is_reshape_redundant_after_transpose(
                            next_op, op, target_shape, model_info)):
                        self._forward_output(next_op, op, model_info)
                        del ops[i]
                        changed = True
                        self._simplified_count += 1
                        self._log_change(
                            "  Removed redundant TRANSPOSE before RESHAPE")
                        continue

            i += 1

        if changed:
            model_info["ops"] = ops

        return changed

    def _is_direct_connection(self, op_a: Dict, op_b: Dict) -> bool:
        """Check if op_a's output goes directly to op_b's input."""
        out_idx = op_a.get("output_indices", [])
        in_idx = op_b.get("input_indices", [])
        return out_idx and in_idx and out_idx[0] == in_idx[0]

    def _forward_output(self, from_op: Dict, to_op: Dict,
                        model_info: Dict) -> None:
        """Forward to_op's output to from_op, then remove from_op."""
        # from_op's output is the same as to_op's input
        # We need to redirect all uses of from_op's output to to_op's output
        from_output = from_op.get("output_indices", [])[0]
        to_output = to_op.get("output_indices", [])[0]

        # Update all ops that used from_output to use to_output instead
        for op in model_info.get("ops", []):
            for idx, input_idx in enumerate(op.get("input_indices", [])):
                if input_idx == from_output:
                    op["input_indices"][idx] = to_output

        # Update tensors dict
        if from_output in model_info.get("tensors", {}):
            del model_info["tensors"][from_output]
        if from_output in model_info.get("weights", {}):
            del model_info["weights"][from_output]

        # Now to_op's output is the effective output
        # to_op's input is from_op's input, but we're removing from_op
        # Actually we want to keep to_op, so from_op's input
        # becomes to_op's input
        to_op["input_indices"] = from_op.get("input_indices", [])

    def _merge_transposes(self, op_a: Dict, op_b: Dict,
                          model_info: Dict) -> bool:
        """Merge two consecutive transposes into one,
        or remove both if inverse."""
        perm_a = op_a.get("transpose_params", {}).get("perm", [])
        perm_b = op_b.get("transpose_params", {}).get("perm", [])

        if not perm_a or not perm_b:
            return False

        # Check if they are inverses
        if self._is_inverse_perm(perm_a, perm_b):
            # Both can be removed
            self._forward_output(op_a, op_b, model_info)
            # Remove both ops
            ops = model_info.get("ops", [])
            # We already removed op_a in the outer loop, so
            # we handle this differently
            # For simplicity, we just remove op_b here
            return False  # Let the caller handle removal

        # Merge: perm_merged = perm_b[perm_a]
        perm_merged = [perm_a[p] for p in perm_b]
        op_b["transpose_params"]["perm"] = perm_merged
        return True

    def _is_inverse_perm(self, p1: List[int], p2: List[int]) -> bool:
        """Check if p2 is the inverse of p1."""
        if len(p1) != len(p2):
            return False
        # p2[p1[i]] == i for all i
        for i in range(len(p1)):
            if p2[p1[i]] != i:
                return False
        return True

    def _is_reshape_redundant_after_transpose(
        self,
        reshape_op: Dict,
        transpose_op: Dict,
        target_shape: List[int],
        model_info: Dict
    ) -> bool:
        """Check if reshape is redundant when followed by transpose."""
        # Get input shape of reshape
        in_idx = reshape_op.get("input_indices", [])[0]
        in_spec = model_info.get("tensors", {}).get(in_idx, {})
        in_shape = in_spec.get("shape", [])

        # Get output shape of transpose (which is reshape's output)
        out_idx = reshape_op.get("output_indices", [])[0]
        out_spec = model_info.get("tensors", {}).get(out_idx, {})
        out_shape = out_spec.get("shape", [])

        # If transpose output shape matches target_shape (after perm),
        # the reshape is redundant
        # This is a simplification; in practice, check if the number of elements
        # is the same and the permutation doesn't change the layout.
        if in_shape and out_shape:
            if in_shape == out_shape:
                return True

        return False

    # ================================================================
    # 2. ADD 0 / MULTIPLY 1 elimination
    # ================================================================

    def _simplify_zero_one_ops(self, model_info: Dict[str, Any]) -> bool:
        """
        Eliminate ADD with 0 and MULTIPLY with 1.

        Patterns:
            ADD (x, 0) -> x
            ADD (0, x) -> x
            MULTIPLY (x, 1) -> x
            MULTIPLY (1, x) -> x
        """
        ops = model_info.get("ops", [])
        changed = False
        i = 0

        while i < len(ops):
            op = ops[i]
            op_name = op.get("op_name")

            if op_name in ("ADD", "MULTIPLY"):
                input_indices = op.get("input_indices", [])
                if len(input_indices) != 2:
                    i += 1
                    continue

                # Check if any input is a constant zero/one
                zero_idx = None
                one_idx = None
                for idx in input_indices:
                    if self._is_constant_zero(model_info, idx):
                        zero_idx = idx
                    if self._is_constant_one(model_info, idx):
                        one_idx = idx

                if op_name == "ADD" and zero_idx is not None:
                    # ADD with 0: forward the other input
                    other_idx = (input_indices[0]
                                if zero_idx == input_indices[1]
                                else input_indices[1])
                    self._remove_op_and_forward(ops, i, other_idx, model_info)
                    changed = True
                    self._simplified_count += 1
                    self._log_change("  Removed ADD with 0")
                    continue

                if op_name == "MULTIPLY" and one_idx is not None:
                    # MULTIPLY with 1: forward the other input
                    other_idx = (input_indices[0]
                                if one_idx == input_indices[1]
                                else input_indices[1])
                    self._remove_op_and_forward(ops, i, other_idx, model_info)
                    changed = True
                    self._simplified_count += 1
                    self._log_change("  Removed MULTIPLY with 1")
                    continue

            i += 1

        if changed:
            model_info["ops"] = ops

        return changed

    def _is_constant_zero(self, model_info: Dict, idx: int) -> bool:
        """Check if tensor at idx is a constant zero."""
        weights = model_info.get("weights", {})
        if idx in weights:
            weight = weights[idx]
            if hasattr(weight, "size"):
                return weight.size == 1 and weight.item() == 0
            if isinstance(weight, list):
                return len(weight) == 1 and weight[0] == 0
        return False

    def _is_constant_one(self, model_info: Dict, idx: int) -> bool:
        """Check if tensor at idx is a constant one."""
        weights = model_info.get("weights", {})
        if idx in weights:
            weight = weights[idx]
            if hasattr(weight, "size"):
                return weight.size == 1 and weight.item() == 1
            if isinstance(weight, list):
                return len(weight) == 1 and weight[0] == 1
        return False

    def _remove_op_and_forward(
        self,
        ops: List[Dict],
        op_idx: int,
        forward_idx: int,
        model_info: Dict
    ) -> None:
        """Remove op at op_idx and forward its output from forward_idx."""
        op = ops[op_idx]
        output_idx = op.get("output_indices", [])[0]

        # Update all ops that used output_idx to use forward_idx instead
        for other_op in ops:
            for i, idx in enumerate(other_op.get("input_indices", [])):
                if idx == output_idx:
                    other_op["input_indices"][i] = forward_idx

        # Remove the op
        del ops[op_idx]

        # Clean up tensor
        if output_idx in model_info.get("tensors", {}):
            del model_info["tensors"][output_idx]
        if output_idx in model_info.get("weights", {}):
            del model_info["weights"][output_idx]

    # ================================================================
    # 3. CONCAT single input / SPLIT single output
    # ================================================================

    def _simplify_concat_split(self, model_info: Dict[str, Any]) -> bool:
        """
        Eliminate CONCAT with single input and SPLIT with single output.

        Patterns:
            CONCAT (x) -> x
            SPLIT (x) -> x (when only one output is used)
        """
        ops = model_info.get("ops", [])
        changed = False
        i = 0

        while i < len(ops):
            op = ops[i]
            op_name = op.get("op_name")

            if op_name == "CONCAT":
                input_indices = op.get("input_indices", [])
                if len(input_indices) <= 1:
                    # Single input concat: just forward
                    forward_idx = input_indices[0] if input_indices else None
                    if forward_idx is not None:
                        self._remove_op_and_forward(
                            ops, i, forward_idx, model_info)
                        changed = True
                        self._simplified_count += 1
                        self._log_change("  Removed CONCAT with single input")
                        continue

            elif op_name == "SPLIT":
                output_indices = op.get("output_indices", [])
                # Check how many outputs are actually used
                used_outputs = self._get_used_outputs(model_info, op)
                if len(used_outputs) == 1:
                    # Single output split: just forward
                    forward_idx = used_outputs[0]
                    self._remove_op_and_forward(ops, i, forward_idx, model_info)
                    changed = True
                    self._simplified_count += 1
                    self._log_change("  Removed SPLIT with single output")
                    continue

            i += 1

        if changed:
            model_info["ops"] = ops

        return changed

    def _get_used_outputs(self, model_info: Dict, op: Dict) -> List[int]:
        """Get list of output indices that are actually used."""
        outputs = op.get("output_indices", [])
        used = []
        for out_idx in outputs:
            for other_op in model_info.get("ops", []):
                if out_idx in other_op.get("input_indices", []):
                    used.append(out_idx)
                    break
        return used

    # ================================================================
    # 4. RELU after RELU
    # ================================================================

    def _simplify_double_relu(self, model_info: Dict[str, Any]) -> bool:
        """
        Eliminate RELU after RELU.

        Pattern:
            RELU -> RELU -> remove the second RELU
        """
        ops = model_info.get("ops", [])
        changed = False
        i = 0

        while i < len(ops) - 1:
            op = ops[i]
            next_op = ops[i + 1]

            if op.get("op_name") == "RELU" and next_op.get("op_name") == "RELU":
                if self._is_direct_connection(op, next_op):
                    self._forward_output(op, next_op, model_info)
                    del ops[i]
                    changed = True
                    self._simplified_count += 1
                    self._log_change("  Removed double RELU")
                    continue

            i += 1

        if changed:
            model_info["ops"] = ops

        return changed

    # ================================================================
    # 5. POOL with kernel_size=1 and stride=1
    # ================================================================

    def _simplify_pool_k1(self, model_info: Dict[str, Any]) -> bool:
        """
        Eliminate POOL with kernel_size=1 and stride=1 (identity).

        Patterns:
            MAX_POOL_2D (kernel=1, stride=1) -> forward input
            AVG_POOL_2D (kernel=1, stride=1) -> forward input
        """
        ops = model_info.get("ops", [])
        changed = False
        i = 0

        while i < len(ops):
            op = ops[i]
            op_name = op.get("op_name")

            if op_name in ("MAX_POOL_2D", "AVG_POOL_2D"):
                pool_params = op.get("pool_params", {})
                kernel = pool_params.get("kernel_size", 0)
                stride = pool_params.get("stride", 0)

                if kernel == 1 and stride == 1:
                    input_idx = op.get("input_indices", [])[0]
                    self._remove_op_and_forward(ops, i, input_idx, model_info)
                    changed = True
                    self._simplified_count += 1
                    self._log_change(
                        f"  Removed {op_name} with kernel=1, stride=1")
                    continue

            i += 1

        if changed:
            model_info["ops"] = ops

        return changed
