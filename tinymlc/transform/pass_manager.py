# tinymlc/transform/pass_manager.py
# Pass manager that runs a sequence of optimization passes.

from typing import Dict, Any, List
from tinymlc.transform.base import Pass
from tinymlc.transform.constant_folding import ConstantFolding
from tinymlc.transform.dce import DeadCodeElimination
from tinymlc.transform.cse import CommonSubexpressionElimination
from tinymlc.transform.simplify import Simplify
from tinymlc.transform.algebraic import AlgebraicSimplify
from tinymlc.transform.fusion import OperatorFusion
from tinymlc.transform.memory import MemoryReuse
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
