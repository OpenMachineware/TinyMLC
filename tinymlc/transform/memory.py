# tinymlc/transform/memory.py
# Memory reuse optimization.

from typing import Dict, Any, List, Set, Tuple
from tinymlc.transform.base import Pass


class MemoryReuse(Pass):
    """
    Memory reuse optimization.

    Reuses memory buffers for tensors whose lifetimes do not overlap.
    This reduces peak RAM usage.
    """

    def __init__(self, name: str = "MemoryReuse"):
        super().__init__(name)
        self._tensor_lifetimes: Dict[int, Tuple[int, int]] = {}  # idx -> (birth, death)
        self._allocation_map: Dict[int, int] = {}  # tensor_idx -> buffer_id

    def run(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run memory reuse on model_info."""
        model_info = self._copy_model(model_info)

        # 1. Compute tensor lifetimes
        self._compute_lifetimes(model_info)

        # 2. Build interference graph
        self._build_allocation(model_info)

        # 3. Assign buffer indices
        self._assign_buffers(model_info)

        return model_info

    def _compute_lifetimes(self, model_info: Dict[str, Any]) -> None:
        """
        Compute birth (first write) and death (last read) for each tensor.

        Birth: when the tensor is first created (as output of some op)
        Death: when the tensor is last used (as input to some op)
        """
        ops = model_info.get("ops", [])

        # Initialize: all tensors from inputs are born at time 0
        # For simplicity, we use op indices as time markers
        birth: Dict[int, int] = {}
        death: Dict[int, int] = {}

        # Input tensors: born at time 0
        for inp in model_info.get("input", []):
            idx = inp.get("tensor_index")
            if idx is not None:
                birth[idx] = 0
                death[idx] = 0

        # Scan ops in order
        for op_idx, op in enumerate(ops):
            # Outputs are born at this op index
            for idx in op.get("output_indices", []):
                birth[idx] = op_idx
                death[idx] = op_idx  # will be updated later

            # Inputs are read at this op index
            for idx in op.get("input_indices", []):
                # If this is the last time this tensor is read, death = op_idx
                # For now, we store death as the latest op that reads it
                death[idx] = op_idx

        # Also handle outputs: they live until the end
        for out in model_info.get("output", []):
            idx = out.get("tensor_index")
            if idx is not None:
                death[idx] = len(ops)  # end of inference

        self._tensor_lifetimes = {
            idx: (birth.get(idx, 0), death.get(idx, 0))
            for idx in set(birth.keys()) | set(death.keys())
        }

        self._log_change(f"Computed lifetimes for {len(self._tensor_lifetimes)} tensors")

    def _build_allocation(self, model_info: Dict[str, Any]) -> None:
        """
        Build allocation map using a greedy algorithm.

        Two tensors can share a buffer if their lifetimes do not overlap.
        """
        tensors = list(self._tensor_lifetimes.keys())
        allocations: Dict[int, int] = {}
        buffer_sizes: Dict[int, int] = {}  # buffer_id -> max size

        # Sort tensors by birth time (earliest first)
        sorted_tensors = sorted(
            tensors,
            key=lambda idx: self._tensor_lifetimes[idx][0]
        )

        for idx in sorted_tensors:
            birth, death = self._tensor_lifetimes[idx]
            size = self._get_tensor_size(model_info, idx)

            # Find an existing buffer that can be reused
            allocated = False
            for buffer_id, buffer_size in buffer_sizes.items():
                # Check if any tensor currently using this buffer overlaps
                overlaps = False
                for allocated_idx, buf_id in allocations.items():
                    if buf_id != buffer_id:
                        continue
                    a_birth, a_death = self._tensor_lifetimes[allocated_idx]
                    if not (death < a_birth or birth > a_death):
                        overlaps = True
                        break

                if not overlaps and size <= buffer_size:
                    allocations[idx] = buffer_id
                    allocated = True
                    break

            if not allocated:
                # Create new buffer
                buffer_id = len(buffer_sizes)
                allocations[idx] = buffer_id
                buffer_sizes[buffer_id] = size

        self._allocation_map = allocations
        self._log_change(f"Allocated {len(buffer_sizes)} buffers for {len(tensors)} tensors")

    def _assign_buffers(self, model_info: Dict[str, Any]) -> None:
        """
        Assign buffer IDs to tensors in model_info.

        This adds a 'buffer_id' field to each tensor spec.
        """
        tensors = model_info.get("tensors", {})
        for idx, spec in tensors.items():
            # Add buffer_id to the tensor spec (will be used by codegen)
            spec.buffer_id = self._allocation_map.get(idx, -1)

        # For JSON output, we also add it to the dict representation
        # But we don't need to expose it to codegen yet

        total_bytes = sum(self._get_tensor_size(model_info, idx) for idx in tensors.keys())
        peak_bytes = sum(
            self._get_tensor_size(model_info, idx)
            for idx, buf_id in self._allocation_map.items()
            if idx in tensors
        )
        self._log_change(f"Peak RAM: {peak_bytes} bytes (total unique: {total_bytes})")

    def _get_tensor_size(self, model_info: Dict[str, Any], idx: int) -> int:
        """Get the size (in bytes) of a tensor."""
        tensors = model_info.get("tensors", {})
        spec = tensors.get(idx)
        if not spec:
            return 0

        shape = spec.shape if hasattr(spec, 'shape') else spec.get('shape', [])
        size = 1
        for d in shape:
            size *= d

        # Assume 1 byte per element for int8
        # TODO: Handle different dtypes
        return size
