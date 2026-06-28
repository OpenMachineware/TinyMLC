# -*- coding: utf-8 -*-
# TinyMLC - Tiny Machine Learning Compiler
#
# Copyright (c) 2026 Jia Liu & TinyMLC Contributors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of TinyMLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Constant folding optimization pass.

from typing import Dict, Any
import numpy as np

from TinyMLC.transform.base import Pass


class ConstantFolding(Pass):
    """
    Constant folding optimization pass.

    This pass evaluates operations at compile time when all inputs
    are constants (known at compile time).

    Currently supports:
        - Reshape with constant shape
        - Transpose with constant permutation
        - Concat with constant axis
        - Add, Multiply, Subtract with constants

    Future extensions:
        - Softmax with constant input
        - Mean with constant axis
    """

    def __init__(self, name: str = "ConstantFolding"):
        super().__init__(name)
        self._const_tensors: Dict[int, np.ndarray] = {}

    def run(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run constant folding on model_info."""
        model_info = self._copy_model(model_info)

        # 1. Find all constant tensors (weights, bias, etc.)
        self._collect_constants(model_info)

        # 2. Scan ops and fold constants
        self._fold_ops(model_info)

        # 3. Prune unused tensors
        self._prune_unused_tensors(model_info)

        return model_info

    def _collect_constants(self, model_info: Dict[str, Any]) -> None:
        """Collect all tensors that are constant (weights, biases, etc.)."""
        weights = model_info.get("weights", {})
        self._const_tensors = {}

        for idx, weight in weights.items():
            if isinstance(weight, np.ndarray):
                self._const_tensors[idx] = weight
            elif isinstance(weight, list):
                self._const_tensors[idx] = np.array(weight)
            else:
                # Scalar or other type
                self._const_tensors[idx] = np.array([weight])

        self._log_change(f"Found {len(self._const_tensors)} constant tensors")

    def _fold_ops(self, model_info: Dict[str, Any]) -> None:
        """Fold operations where all inputs are constants."""
        ops = model_info.get("ops", [])
        new_ops = []
        folded_count = 0

        for op in ops:
            op_name = op.get("op_name")
            folded = False

            # Only fold if we can evaluate it now
            if op_name == "RESHAPE":
                folded = self._fold_reshape(model_info, op)
            elif op_name == "TRANSPOSE":
                folded = self._fold_transpose(model_info, op)
            elif op_name in ("ADD", "MULTIPLY", "SUB"):
                folded = self._fold_binary_op(model_info, op)
            elif op_name == "MEAN":
                folded = self._fold_mean(model_info, op)

            if folded:
                folded_count += 1
                self._log_change(f"Folded {op_name}")
            else:
                new_ops.append(op)

        if folded_count > 0:
            model_info["ops"] = new_ops
            self._log_change(f"Folded {folded_count} ops")

    def _fold_reshape(
        self, model_info: Dict[str, Any], op: Dict[str, Any]
    ) -> bool:
        """Fold reshape if input is constant."""
        input_idx = op.get("input_indices", [])[0]
        output_idx = op.get("output_indices", [])[0]

        if input_idx in self._const_tensors:
            try:
                # Get the target shape from params
                params = op.get("reshape_params", {})
                target_shape = params.get("shape", [])
                if not target_shape:
                    target_shape = params.get("target_shape", [])

                data = self._const_tensors[input_idx]
                folded = data.reshape(target_shape)

                # Store as constant tensor
                self._const_tensors[output_idx] = folded

                # Add to weights so it gets written out
                model_info["weights"][output_idx] = folded

                self._log_change(
                    f"  Reshape constant: {data.shape} -> {folded.shape}"
                )
                return True
            except Exception as e:
                print(f"  Warning: failed to fold reshape: {e}")
                return False
        return False

    def _fold_transpose(
        self, model_info: Dict[str, Any], op: Dict[str, Any]
    ) -> bool:
        """Fold transpose if input is constant."""
        input_idx = op.get("input_indices", [])[0]
        output_idx = op.get("output_indices", [])[0]

        if input_idx in self._const_tensors:
            try:
                params = op.get("transpose_params", {})
                perm = params.get("perm", [])

                data = self._const_tensors[input_idx]
                folded = np.transpose(data, axes=perm or None)

                self._const_tensors[output_idx] = folded
                model_info["weights"][output_idx] = folded

                self._log_change(
                    f"  Transpose constant: {data.shape} -> {folded.shape}"
                )
                return True
            except Exception as e:
                print(f"  Warning: failed to fold transpose: {e}")
                return False
        return False

    def _fold_binary_op(
        self, model_info: Dict[str, Any], op: Dict[str, Any]
    ) -> bool:
        """Fold binary ops (ADD, MULTIPLY, SUB) if all inputs constant."""
        op_name = op.get("op_name")
        input_indices = op.get("input_indices", [])
        output_idx = op.get("output_indices", [0])[0]

        if all(idx in self._const_tensors for idx in input_indices):
            try:
                a = self._const_tensors[input_indices[0]]
                b = self._const_tensors[input_indices[1]]

                if op_name == "ADD":
                    folded = a + b
                elif op_name == "MULTIPLY":
                    folded = a * b
                elif op_name == "SUB":
                    folded = a - b
                else:
                    return False

                self._const_tensors[output_idx] = folded
                model_info["weights"][output_idx] = folded

                self._log_change(
                    f"  {op_name} constant: {a.shape} + {b.shape} "
                    f"-> {folded.shape}"
                )
                return True
            except Exception as e:
                print(f"  Warning: failed to fold {op_name}: {e}")
                return False
        return False

    def _fold_mean(
        self, model_info: Dict[str, Any], op: Dict[str, Any]
    ) -> bool:
        """Fold MEAN if input is constant."""
        input_idx = op.get("input_indices", [0])[0]
        output_idx = op.get("output_indices", [0])[0]

        if input_idx in self._const_tensors:
            try:
                params = op.get("mean_params", {})
                axis = params.get("axis", None)
                keepdims = params.get("keepdims", False)

                data = self._const_tensors[input_idx]
                folded = np.mean(data, axis=axis, keepdims=keepdims)

                self._const_tensors[output_idx] = folded
                model_info["weights"][output_idx] = folded

                self._log_change(
                    f"  Mean constant: {data.shape} -> {folded.shape}"
                )
                return True
            except Exception as e:
                print(f"  Warning: failed to fold mean: {e}")
                return False
        return False

    def _prune_unused_tensors(self, model_info: Dict[str, Any]) -> None:
        """Remove tensors that are no longer used."""
        # Get all used tensor indices from ops
        used_indices = set()
        for op in model_info.get("ops", []):
            for idx in op.get("input_indices", []):
                used_indices.add(idx)
            for idx in op.get("output_indices", []):
                used_indices.add(idx)

        # Get input/output indices
        for inp in model_info.get("input", []):
            # Inputs don't have indices in this representation
            pass

        # Remove unused tensors
        all_indices = set(model_info["tensors"].keys())
        unused = all_indices - used_indices

        for idx in unused:
            if idx in model_info["tensors"]:
                del model_info["tensors"][idx]
            if idx in model_info["weights"]:
                del model_info["weights"][idx]

        if unused:
            self._log_change(f"Removed {len(unused)} unused tensors")
