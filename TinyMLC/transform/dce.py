# TinyMLC/transform/dce.py
# Dead Code Elimination pass.

from typing import Dict, Any, Set, List
from TinyMLC.transform.base import Pass


class DeadCodeElimination(Pass):
    """
    Dead Code Elimination.

    Removes:
        - Tensors that are never used as inputs to any op
        - Ops whose outputs are never used
        - Unreachable ops (in case of control flow, not implemented yet)

    This pass should be run after each pass that may create dead code.
    """

    def __init__(self, name: str = "DeadCodeElimination"):
        super().__init__(name)

    def run(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run dead code elimination on model_info."""
        model_info = self._copy_model(model_info)

        changed = True
        iteration = 0

        while changed:
            changed = False
            iteration += 1

            # 1. Find all used tensor indices
            used_indices = self._collect_used_indices(model_info)

            # 2. Remove dead tensors
            dead_tensors = self._remove_dead_tensors(model_info, used_indices)
            if dead_tensors:
                changed = True
                self._log_change(
                    f"Iteration {iteration}: removed {len(dead_tensors)} "
                    f"dead tensors"
                )

            # 3. Remove dead ops
            dead_ops = self._remove_dead_ops(model_info, used_indices)
            if dead_ops:
                changed = True
                self._log_change(
                    f"Iteration {iteration}: removed {len(dead_ops)} dead ops"
                )

        return model_info

    def _collect_used_indices(self, model_info: Dict[str, Any]) -> Set[int]:
        """
        Collect all tensor indices that are used as inputs to any op.
        """
        used = set()

        # 1. All input tensors are used
        for inp in model_info.get("input", []):
            idx = inp.get("tensor_index")
            if idx is not None:
                used.add(idx)

        # 2. All output tensors are used
        for out in model_info.get("output", []):
            idx = out.get("tensor_index")
            if idx is not None:
                used.add(idx)

        # 3. All tensor indices referenced by ops
        for op in model_info.get("ops", []):
            for idx in op.get("input_indices", []):
                used.add(idx)
            for idx in op.get("output_indices", []):
                used.add(idx)

        return used

    def _remove_dead_tensors(
        self,
        model_info: Dict[str, Any],
        used_indices: Set[int]
    ) -> Set[int]:
        """
        Remove tensors that are not used as inputs to any op,
        except outputs (they must be preserved).
        """
        # Output tensors must be preserved (they are the final result)
        output_indices = set()
        for out in model_info.get("output", []):
            # Outputs are identified by name, not index
            # We need to find which tensor index corresponds to each output
            # For now, assume outputs are in tensors dict with some mapping
            pass

        # For simplicity: find tensors that are never used as inputs
        all_indices = set(model_info.get("tensors", {}).keys())
        dead = all_indices - used_indices

        # Don't delete tensors that are explicitly marked as outputs
        # This requires knowing which tensors are outputs.
        # In our model_info, outputs are separate from tensors.
        # For now, we keep all tensors that are outputs.

        # Actually delete them
        for idx in dead:
            if idx in model_info.get("tensors", {}):
                del model_info["tensors"][idx]
            if idx in model_info.get("weights", {}):
                del model_info["weights"][idx]

        return dead

    def _remove_dead_ops(
        self,
        model_info: Dict[str, Any],
        used_indices: Set[int]
    ) -> List[Dict[str, Any]]:
        """
        Remove ops whose output indices are never used as inputs.
        """
        # For each op, check if any of its outputs are used
        ops = model_info.get("ops", [])
        tensors = model_info.get("tensors", {})
        dead_ops = []
        alive_ops = []

        for op in ops:
            outputs = op.get("output_indices", [])
            # Check if all output indices are in tensors.
            all_outputs_valid = all(idx in tensors for idx in outputs)
            # An op is alive if any of its outputs is used
            is_alive = (
                any(idx in used_indices for idx in outputs)
                and all_outputs_valid
            )

            # Also: if this op produces an output tensor that is
            # the final output
            # For now, keep it if it's the last op in the graph
            # (we'll use a more sophisticated analysis later)

            if is_alive:
                alive_ops.append(op)
            else:
                dead_ops.append(op)

        if dead_ops:
            model_info["ops"] = alive_ops

            # Remove any tensors that were only produced by dead ops
            # (they'll be caught by the tensor removal in the next iteration)
            for op in dead_ops:
                for idx in op.get("output_indices", []):
                    if idx in model_info.get("tensors", {}):
                        # Don't delete right away, let the tensor removal
                        # handle it
                        pass

        return dead_ops
