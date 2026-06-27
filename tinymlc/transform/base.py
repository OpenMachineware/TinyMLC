# tinymlc/transform/base.py
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
