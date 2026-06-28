# TinyMLC/transform/pass_manager.py
# Pass manager that runs a sequence of optimization passes.

import json
import sys

from typing import Dict, Any, List
from TinyMLC.transform.base import Pass
from TinyMLC.transform.constant_folding import ConstantFolding
from TinyMLC.transform.dce import DeadCodeElimination
from TinyMLC.transform.cse import CommonSubexpressionElimination
from TinyMLC.transform.simplify import Simplify
from TinyMLC.transform.algebraic import AlgebraicSimplify
from TinyMLC.transform.fusion import OperatorFusion
from TinyMLC.transform.memory import MemoryReuse
from utils.dump import info


class PassManager:
    """
    Manages and runs a sequence of optimization passes.
    """

    def __init__(self, passes: List[Pass] = None):
        self.passes = passes or []
        self._results = []

    def add_pass(self, pass_obj: Pass) -> None:
        """Add a pass to the pipeline."""
        self.passes.append(pass_obj)

    def run(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all passes in sequence on model_info.

        Returns:
            The transformed model_info after all passes.
        """
        current = model_info
        self._results = []

        for pass_obj in self.passes:
            info(f"  Running: {pass_obj.name}")
            current = pass_obj.run(current)
            print(f"OPTIMIZED_MODEL: {json.dumps(current)}")
            sys.stdout.flush()
            self._results.append({
                "name": pass_obj.name,
                "stats": pass_obj.get_stats(),
            })

        return current

    @classmethod
    def default_pipeline(cls) -> "PassManager":
        """Create the default optimization pass pipeline."""
        pm = cls()
        pm.add_pass(ConstantFolding())
        pm.add_pass(DeadCodeElimination())
        pm.add_pass(CommonSubexpressionElimination())
        pm.add_pass(DeadCodeElimination())
        pm.add_pass(Simplify())
        pm.add_pass(DeadCodeElimination())
        pm.add_pass(OperatorFusion())
        pm.add_pass(DeadCodeElimination())
        pm.add_pass(AlgebraicSimplify())
        pm.add_pass(DeadCodeElimination())
        pm.add_pass(MemoryReuse())
        return pm

    def get_results(self) -> List[Dict[str, Any]]:
        """Get statistics from all passes."""
        return self._results

    def dump_summary(self) -> None:
        """Dump a summary of all passes."""
        info("\n=== Pass Summary ===")
        for result in self._results:
            stats = result["stats"]
            changes = stats.get("changes", [])
            info(f"  {result['name']}: {len(changes)} changes")
            for change in changes:
                info(f"    - {change}")
