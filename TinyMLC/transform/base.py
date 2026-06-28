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

# Base class for all optimization passes.

from abc import ABC, abstractmethod
from typing import Dict, Any
import copy


class Pass(ABC):
    """
    Base class for all optimization passes.

    Each pass takes a model_info dict, transforms it, and returns
    the transformed model_info.
    """

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self._stats = {
            "before": {},
            "after": {},
            "changes": [],
        }

    @abstractmethod
    def run(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run the pass on model_info and return transformed model_info."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the pass execution."""
        return self._stats

    def _log_change(self, msg: str) -> None:
        """Record a change made by this pass."""
        self._stats["changes"].append(msg)

    def _copy_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy model_info to avoid mutating the original."""
        return copy.deepcopy(model_info)
