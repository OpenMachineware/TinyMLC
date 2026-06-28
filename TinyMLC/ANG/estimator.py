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

# Abstract base class for all estimators.

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from TinyMLC.ANG.utils import hash_structure

class Estimator(ABC):
    """
    Abstract base class for network performance estimation.

    All estimators (Software, QEMU, Hardware HAL) inherit from this class
    and implement the estimate() method.

    The estimate() method takes a model_info structure and returns
    a Score dictionary with performance metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the estimator with optional configuration.

        Args:
            config: Configuration dictionary for the estimator.
        """
        self.config = config or {}
        self._cache: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def estimate(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate the performance of a given model.

        The returned dictionary must contain at least:
            - 'score': float, higher is better
            - 'macs': int
            - 'params': int
            - 'peak_ram': int (bytes)
            - 'flash': int (bytes)
            - 'latency_ms': float

        Additional fields can be added as needed.

        Args:
            model_info: ModelInfo dictionary.

        Returns:
            Dictionary with performance metrics.
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, str]:
        """
        Get information about this estimator.

        Returns:
            Dictionary with estimator name, version, and description.
        """
        pass

    def estimate_cached(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate with caching to avoid redundant evaluations.

        Args:
            model_info: ModelInfo dictionary.

        Returns:
            Dictionary with performance metrics.
        """
        # Compute a hash of the model structure
        key = hash_structure(model_info)

        if key in self._cache:
            return self._cache[key]

        result = self.estimate(model_info)
        self._cache[key] = result
        return result

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()
