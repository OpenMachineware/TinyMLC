# TinyMLC/transform/cse.py
# Common Subexpression Elimination.

from typing import Dict, Any
import hashlib
import json

from TinyMLC.transform.base import Pass


class CommonSubexpressionElimination(Pass):
    """
    Common Subexpression Elimination.

    Finds and eliminates duplicate computations:
        - Same op with same inputs and same params
        - Same constant tensor being computed multiple times

    Strategy:
        1. Compute a signature for each op (op_name + input_indices + params)
        2. If two ops have the same signature, keep the first one
        3. Replace all uses of the later op's outputs with
           the first op's outputs
    """

    def __init__(self, name: str = "CommonSubexpressionElimination"):
        super().__init__(name)
        self._signature_map: Dict[str, int] = {}  # signature -> op_index
        self._replace_map: Dict[int, int] = {}    # old_tensor -> new_tensor

    def run(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run CSE on model_info."""
        model_info = self._copy_model(model_info)

        self._signature_map.clear()
        self._replace_map.clear()

        ops = model_info.get("ops", [])
        new_ops = []
        eliminated_count = 0

        for op in ops:
            signature = self._compute_signature(op)

            if signature in self._signature_map:
                # Duplicate found: replace outputs with the original
                orig_op_idx = self._signature_map[signature]
                orig_op = new_ops[orig_op_idx]
                self._replace_outputs(op, orig_op)
                eliminated_count += 1
                self._log_change(
                    f"  Eliminated duplicate {op.get('op_name')} "
                    f"(outputs: {op.get('output_indices')} "
                    f"-> {orig_op.get('output_indices')})"
                )
                # Don't add this op to new_ops
            else:
                # New op: record its signature and keep it
                self._signature_map[signature] = len(new_ops)
                new_ops.append(op)

        if eliminated_count > 0:
            model_info["ops"] = new_ops
            # Update all tensor references in remaining ops
            self._update_tensor_refs(model_info)
            self._log_change(
                f"Eliminated {eliminated_count} duplicate expressions"
            )

        return model_info

    def _compute_signature(self, op: Dict[str, Any]) -> str:
        """
        Compute a unique signature for an op.

        The signature includes:
            - op_name
            - input_indices (sorted for commutativity)
            - output_indices (for ops with multiple outputs)
            - params (sorted, excluding irrelevant fields)
        """
        op_name = op.get("op_name", "UNKNOWN")

        # Input indices: sorted for commutative ops? Not always safe.
        # For now, keep the order as-is, as ops like SUB are not commutative.
        input_indices = op.get("input_indices", [])

        # Params: filter out fields that don't affect computation
        params = op.get("params", {})
        # Remove fields that are just metadata
        skip_keys = {"name", "index", "state", "pass_flags"}
        filtered_params = {
            k: v for k, v in params.items()
            if k not in skip_keys and not k.startswith("_")
        }

        # Build signature dict
        sig = {
            "op_name": op_name,
            "input_indices": input_indices,
            "params": filtered_params,
        }

        # Hash to a string
        sig_str = json.dumps(sig, sort_keys=True)
        return hashlib.sha256(sig_str.encode()).hexdigest()[:16]

    def _replace_outputs(
        self, dup_op: Dict[str, Any], orig_op: Dict[str, Any]
    ) -> None:
        """
        Map outputs of dup_op to outputs of orig_op.

        Assumes the output indices are in the same order.
        """
        dup_outputs = dup_op.get("output_indices", [])
        orig_outputs = orig_op.get("output_indices", [])

        if len(dup_outputs) != len(orig_outputs):
            # Different number of outputs, can't replace
            self._log_change(
                f"  Warning: output count mismatch "
                f"({len(dup_outputs)} vs {len(orig_outputs)})"
            )
            return

        for dup_idx, orig_idx in zip(dup_outputs, orig_outputs):
            self._replace_map[dup_idx] = orig_idx

    def _update_tensor_refs(self, model_info: Dict[str, Any]) -> None:
        """
        Update all tensor references in ops:
            - Replace old tensor indices with new ones
            - Remove any ops that now have duplicate inputs/outputs
        """
        if not self._replace_map:
            return

        ops = model_info.get("ops", [])

        for op in ops:
            # Update input_indices
            input_indices = op.get("input_indices", [])
            new_inputs = [
                self._replace_map.get(idx, idx) for idx in input_indices
            ]
            op["input_indices"] = new_inputs

            # Update output_indices
            output_indices = op.get("output_indices", [])
            new_outputs = [
                self._replace_map.get(idx, idx) for idx in output_indices
            ]
            op["output_indices"] = new_outputs

        # Update tensors dict: remove replaced tensors
        tensors = model_info.get("tensors", {})
        for old_idx in self._replace_map.keys():
            if old_idx in tensors:
                del tensors[old_idx]

        # Update weights dict
        weights = model_info.get("weights", {})
        for old_idx in self._replace_map.keys():
            if old_idx in weights:
                del weights[old_idx]

        # Update tensor references in input/output specs
        # (tensor_index is metadata, we don't need to update it for CSE)
        # But if we want to keep consistency, we could update it.

        self._log_change(
            f"  Replaced {len(self._replace_map)} tensor references"
        )
