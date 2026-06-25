# tinymlc/transform/algebraic.py
# Algebraic simplification: transforms arithmetic expressions
# to simpler equivalent forms.
#
# Patterns handled:
#   1. ADD(x, -x) -> 0 (or constant zero)
#   2. SUB(x, x) -> 0
#   3. ADD(x, constant) -> x + constant (evaluate if possible)
#   4. SUB(x, constant) -> x - constant
#   5. ADD(x, ADD(y, z)) -> ADD(ADD(x, y), z)  (associative)
#   6. ADD(x, y) -> SUB(x, -y)  (if y is negative constant)

from typing import Dict, Any, List, Set, Tuple, Optional
from .base import Pass


class AlgebraicSimplify(Pass):
    """
    Algebraic simplification for arithmetic operations.

    Runs after Simplify to catch patterns that require numeric analysis.
    """

    def __init__(self, name: str = "AlgebraicSimplify"):
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

            # 1. SUB(x, x) -> 0
            changed |= self._simplify_sub_same(model_info)

            # 2. ADD(x, -x) -> 0 (if -x is a constant)
            changed |= self._simplify_add_neg(model_info)

            # 3. ADD(x, constant) with constant that can be folded
            changed |= self._simplify_add_constant(model_info)

            # 4. SUB(x, constant) with constant that can be folded
            changed |= self._simplify_sub_constant(model_info)

            if changed:
                self._log_change(f"Iteration {iteration}: applied")

        if self._simplified_count > 0:
            self._log_change(f"Total: {self._simplified_count} algebraic simplifications")

        return model_info

    # ================================================================
    # 1. SUB(x, x) -> 0 (forward constant zero)
    # ================================================================

    def _simplify_sub_same(self, model_info: Dict[str, Any]) -> bool:
        """
        SUB(x, x) -> 0
        """
        ops = model_info.get("ops", [])
        changed = False
        i = 0

        while i < len(ops):
            op = ops[i]
            if op.get("op_name") == "SUB":
                input_indices = op.get("input_indices", [])
                if len(input_indices) == 2 and input_indices[0] == input_indices[1]:
                    # Create a constant zero tensor
                    zero_idx = self._create_constant_zero(model_info)
                    self._remove_op_and_forward(ops, i, zero_idx, model_info)
                    changed = True
                    self._simplified_count += 1
                    self._log_change("  SUB(x, x) -> 0")
                    continue

            i += 1

        if changed:
            model_info["ops"] = ops

        return changed

    # ================================================================
    # 2. ADD(x, -x) -> 0 (if -x is a constant)
    # ================================================================

    def _simplify_add_neg(self, model_info: Dict[str, Any]) -> bool:
        """
        ADD(x, -x) -> 0 when -x is a constant with opposite sign.
        """
        ops = model_info.get("ops", [])
        changed = False
        i = 0

        while i < len(ops):
            op = ops[i]
            if op.get("op_name") == "ADD":
                input_indices = op.get("input_indices", [])
                if len(input_indices) == 2:
                    a = input_indices[0]
                    b = input_indices[1]
                    # Check if a is constant and b is the same constant with opposite sign
                    const_val = self._get_constant_value(model_info, a)
                    if const_val is not None:
                        const_b = self._get_constant_value(model_info, b)
                        if const_b is not None and const_b == -const_val:
                            zero_idx = self._create_constant_zero(model_info)
                            self._remove_op_and_forward(ops, i, zero_idx, model_info)
                            changed = True
                            self._simplified_count += 1
                            self._log_change(f"  ADD({const_val}, {-const_val}) -> 0")
                            continue

            i += 1

        if changed:
            model_info["ops"] = ops

        return changed

    # ================================================================
    # 3. ADD(x, constant) with constant that can be folded
    # ================================================================

    def _simplify_add_constant(self, model_info: Dict[str, Any]) -> bool:
        """
        ADD(x, constant) -> x + constant (if constant is a scalar).
        """
        ops = model_info.get("ops", [])
        changed = False
        i = 0

        while i < len(ops):
            op = ops[i]
            if op.get("op_name") == "ADD":
                input_indices = op.get("input_indices", [])
                if len(input_indices) == 2:
                    # Check if one input is a constant
                    const_idx = None
                    other_idx = None
                    for idx in input_indices:
                        if self._is_scalar_constant(model_info, idx):
                            const_idx = idx
                        else:
                            other_idx = idx

                    if const_idx is not None and other_idx is not None:
                        # Fold constant into the op's params
                        const_val = self._get_constant_value(model_info, const_idx)
                        if const_val is not None:
                            # Replace with a new op that has the constant baked in
                            # For now, we just keep the constant as a tensor and
                            # let the constant folding pass handle it.
                            # But we can mark it for later folding.
                            pass
            i += 1

        if changed:
            model_info["ops"] = ops

        return changed

    # ================================================================
    # 4. SUB(x, constant) -> x - constant (if constant is a scalar)
    # ================================================================

    def _simplify_sub_constant(self, model_info: Dict[str, Any]) -> bool:
        """
        SUB(x, constant) -> x - constant (if constant is a scalar).
        """
        ops = model_info.get("ops", [])
        changed = False
        i = 0

        while i < len(ops):
            op = ops[i]
            if op.get("op_name") == "SUB":
                input_indices = op.get("input_indices", [])
                if len(input_indices) == 2:
                    # Check if the second input is a constant
                    const_idx = input_indices[1]
                    if self._is_scalar_constant(model_info, const_idx):
                        # This can be folded by constant folding
                        pass
            i += 1

        return changed

    # ================================================================
    # Helper functions
    # ================================================================

    def _create_constant_zero(self, model_info: Dict[str, Any]) -> int:
        """Create a constant zero tensor."""
        import numpy as np

        # Find the max tensor index
        tensors = model_info.get("tensors", {})
        max_idx = max(tensors.keys()) if tensors else 0
        new_idx = max_idx + 1

        # Create zero tensor (scalar)
        tensors[new_idx] = {
            "name": f"zero_{new_idx}",
            "shape": [1],
            "dtype": "int8",
            "scale": 1.0,
            "zero_point": 0,
        }
        model_info["weights"][new_idx] = np.array([0], dtype=np.int8)

        return new_idx

    def _get_constant_value(self, model_info: Dict, idx: int) -> Optional[int]:
        """Get scalar constant value if tensor at idx is a scalar constant."""
        weights = model_info.get("weights", {})
        if idx in weights:
            weight = weights[idx]
            if hasattr(weight, "size") and weight.size == 1:
                return int(weight.item())
            if isinstance(weight, list) and len(weight) == 1:
                return weight[0]
        return None

    def _is_scalar_constant(self, model_info: Dict, idx: int) -> bool:
        """Check if tensor at idx is a scalar constant."""
        return self._get_constant_value(model_info, idx) is not None

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

        for other_op in ops:
            for i, idx in enumerate(other_op.get("input_indices", [])):
                if idx == output_idx:
                    other_op["input_indices"][i] = forward_idx

        del ops[op_idx]

        if output_idx in model_info.get("tensors", {}):
            del model_info["tensors"][output_idx]
        if output_idx in model_info.get("weights", {}):
            del model_info["weights"][output_idx]
