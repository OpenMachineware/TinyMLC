# model_generator.py
# Network structure generation using random and genetic algorithms.

import random
import copy
from typing import Dict, Any, List, Optional

from tinymlc.ANG.model_builder import ModelBuilder
from tinymlc.ANG.model_info import ModelInfo
from tinymlc.ANG.estimator import Estimator
from tinymlc.ANG.utils import (generate_random_weights_from_structure,
                               fill_model_info_with_weights)
from utils.dump import fatal_error, warning, info


class ModelGenerator:
    """
    Model structure generator.

    This class handles:
        - Random: Generate random networks and pick the best.
        - Genetic algorithm: Iteratively evolve a population of networks.
    """

    def __init__(
        self,
        estimator: Estimator,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.estimator = estimator
        self.config = config or {}

        self.default_config = {
            # params
            "population_size": 50,
            "generations": 50,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "tournament_size": 3,
            "early_stop": 10,
            # Network structure params
            "max_layers": 10,
            "min_layers": 3,
            "channels_options": [8, 16, 32, 64],
            "kernel_options": [1, 3, 5],
            "stride_options": [1, 2],
            "fc_units_options": [32, 64, 128],
            "upsample_factors": [2, 4],
            # Task config
            "task_type": "classification",      # classification / detection / segmentation
            "input_shape": [1, 28, 28, 1],
            "output_shape": [1, 10],
            # Constraints (for scoring)
            "max_macs": 100000,
            "max_ram": 30 * 1024,
            "max_flash": 64 * 1024,
        }
        self.config = {**self.default_config, **(config or {})}
        self._best_score = -1.0
        self._best_structure = None

    def _is_image_input(self) -> bool:
        """Check if input is image-like (4D: NHWC)."""
        shape = self.config["input_shape"]
        return len(shape) == 4

    def _is_1d_input(self) -> bool:
        """Check if input is 1D (sensor, audio, etc)."""
        shape = self.config["input_shape"]
        return len(shape) == 2 or (len(shape) == 3 and shape[0] == 1)

    def generate_random(self, num_samples: int = 100) -> Dict[str, Any]:
        """Random."""
        best_structure = None
        best_score = -1.0

        for _ in range(num_samples):
            structure = self._generate_random_structure()
            model_info = self._structure_to_model_info(structure)
            result = self.estimator.estimate(model_info)
            score = result.get("score", 0.0)

            if score > best_score:
                best_score = score
                best_structure = structure

        self._best_score = best_score
        self._best_structure = best_structure
        return best_structure

    def generate_evolved(self) -> Dict[str, Any]:
        """Genetic algorithm."""
        population_size = self.config["population_size"]
        population = [
            self._generate_random_structure()
            for _ in range(population_size)
        ]

        best_structure = None
        best_score = -1.0
        no_improvement_count = 0
        early_stop = self.config["early_stop"]

        for generation in range(self.config["generations"]):
            scores = []
            for structure in population:
                model_info = self._structure_to_model_info(structure)
                result = self.estimator.estimate(model_info)
                scores.append(result.get("score", 0.0))

            gen_best_score = max(scores)
            gen_best_idx = scores.index(gen_best_score)

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_structure = copy.deepcopy(population[gen_best_idx])
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stop:
                break

            selected = self._tournament_selection(population, scores)

            next_population = []
            for i in range(0, population_size, 2):
                parent1 = selected[i % len(selected)]
                parent2 = selected[(i + 1) % len(selected)]

                if random.random() < self.config["crossover_rate"]:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                next_population.append(child1)
                if len(next_population) < population_size:
                    next_population.append(child2)

            population = next_population

        self._best_structure = best_structure
        return best_structure

    def _generate_random_structure(self) -> Dict[str, Any]:
        """Generate a random network structure based on task type."""
        task_type = self.config["task_type"]
        max_layers = self.config["max_layers"]
        min_layers = self.config["min_layers"]
        num_layers = random.randint(min_layers, max_layers)

        layers = []
        input_shape = self.config["input_shape"]
        output_shape = self.config["output_shape"]

        for i in range(num_layers):
            # Last layer depends on task type
            if i == num_layers - 1:
                if task_type == "classification":
                    layer_type = "fc"
                elif task_type == "detection":
                    layer_type = "detection_head"
                else:  # segmentation
                    layer_type = "conv"  # final conv for pixel-wise output
            elif i == 0:
                # First layer is always conv
                layer_type = "conv"
            else:
                # Middle layers: mix of conv, pool
                if task_type == "segmentation":
                    # Segmentation needs upsample in decoder
                    # For now, keep it simple with conv only (pool not implemented)
                    layer_type = "conv"
                else:
                    layer_type = "conv"
                    # TODO: Re-enable pool layer after implementing builder.add_pool()
                    # layer_type = random.choice(["conv", "pool"])

            if layer_type == "conv":
                channels = random.choice(self.config["channels_options"])
                kernel = random.choice(self.config["kernel_options"])
                stride = random.choice(self.config["stride_options"])
                layers.append({
                    "type": "conv",
                    "channels": channels,
                    "kernel": kernel,
                    "stride": stride,
                })
            elif layer_type == "pool":
                kernel = random.choice([2, 3])
                stride = random.choice([2, 3])
                layers.append({
                    "type": "pool",
                    "kernel": kernel,
                    "stride": stride,
                })
            elif layer_type == "fc":
                units = random.choice(self.config["fc_units_options"])
                layers.append({
                    "type": "fc",
                    "units": units,
                })
            elif layer_type == "detection_head":
                # Detection head: a few conv layers + output
                layers.append({
                    "type": "detection_head",
                    "num_anchors": random.choice([3, 4, 5]),
                })

        return {
            "layers": layers,
            "input_shape": input_shape,
            "output_shape": output_shape,
        }

    def _structure_to_model_info(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Convert structure to model_info without weights."""
        builder = ModelBuilder("ang_generated")

        input_shape = structure["input_shape"]
        output_shape = structure["output_shape"]

        input_idx = builder.add_input("input", input_shape, "int8")

        current_idx = input_idx
        current_shape = list(input_shape)

        for layer in structure["layers"]:
            layer_type = layer["type"]

            if layer_type == "conv":
                kernel = layer.get("kernel", 3)
                channels = layer.get("channels", 16)
                stride = layer.get("stride", 1)

                # Build output shape based on input dimensions
                if len(current_shape) == 4:  # NHWC image
                    output_shape_layer = list(current_shape)
                    output_shape_layer[-1] = channels
                elif len(current_shape) == 3:  # 1D with channel
                    output_shape_layer = list(current_shape)
                    output_shape_layer[-1] = channels
                else:
                    # Fallback: just append channels
                    output_shape_layer = current_shape + [channels]

                output_idx = builder.add_tensor("conv_out",
                                                output_shape_layer, "int8")
                builder.add_conv(current_idx, output_idx,
                                 kernel, channels, stride)
                current_idx = output_idx
                current_shape = output_shape_layer

            elif layer_type == "pool":
                kernel = layer.get("kernel", 2)
                stride = layer.get("stride", 2)

                # Pool reduces spatial dimensions
                if len(current_shape) == 4:  # NHWC
                    output_shape_layer = list(current_shape)
                    output_shape_layer[1] = max(1, output_shape_layer[1] // stride)
                    output_shape_layer[2] = max(1, output_shape_layer[2] // stride)
                elif len(current_shape) == 3:
                    output_shape_layer = list(current_shape)
                    output_shape_layer[1] = max(1, output_shape_layer[1] // stride)
                else:
                    output_shape_layer = current_shape

                output_idx = builder.add_tensor("pool_out", output_shape_layer, "int8")
                # TODO: Add pool op to builder
                current_idx = output_idx
                current_shape = output_shape_layer

            elif layer_type == "fc":
                units = layer.get("units", 64)
                output_idx = builder.add_tensor("fc_out", [1, units], "int8")
                builder.add_fc(current_idx, output_idx, units)
                current_idx = output_idx
                current_shape = [1, units]

            elif layer_type == "detection_head":
                # Detection head: placeholder
                num_anchors = layer.get("num_anchors", 3)
                num_classes = output_shape[-1] if len(output_shape) > 1 else 10
                # Output boxes: [N, num_anchors, 4]
                # Output classes: [N, num_anchors, num_classes]
                boxes_idx = builder.add_tensor(
                    "detection_boxes", [1, num_anchors, 4], "int8"
                )
                classes_idx = builder.add_tensor(
                    "detection_classes", [1, num_anchors, num_classes], "int8"
                )
                builder.add_detection_head(
                    current_idx, boxes_idx, classes_idx,
                    num_anchors=num_anchors, num_classes=num_classes
                )
                current_idx = boxes_idx  # Just use boxes as output ref
                current_shape = [1, num_anchors, 4]

        # Final output
        output_idx = builder.add_output("output", output_shape, "int8")
        if current_idx != output_idx:
            # If current shape doesn't match output_shape, add a final FC
            flat_size = 1
            for d in current_shape:
                flat_size *= d
            builder.add_fc(current_idx, output_idx, output_shape[-1], "none")

        model_info = builder.build()
        return model_info.to_dict()

    def _tournament_selection(self, population, scores):
        """Tournament selection."""
        selected = []
        tournament_size = self.config["tournament_size"]

        for _ in range(len(population)):
            indices = random.sample(range(len(population)), tournament_size)
            best_idx = max(indices, key=lambda i: scores[i])
            selected.append(population[best_idx])

        return selected

    def _crossover(self, parent1, parent2):
        """Single-point crossover."""
        layers1 = parent1["layers"]
        layers2 = parent2["layers"]

        if len(layers1) < 2 or len(layers2) < 2:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        pos1 = random.randint(1, len(layers1) - 1)
        pos2 = random.randint(1, len(layers2) - 1)

        child1_layers = layers1[:pos1] + layers2[pos2:]
        child2_layers = layers2[:pos2] + layers1[pos1:]

        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        child1["layers"] = child1_layers
        child2["layers"] = child2_layers

        return child1, child2

    def _mutate(self, structure):
        """Mutate a structure."""
        layers = structure["layers"]

        if not layers:
            return structure

        mutation_rate = self.config["mutation_rate"]

        for i, layer in enumerate(layers):
            if random.random() < mutation_rate:
                layer_type = layer["type"]
                if layer_type == "conv":
                    layer["channels"] = random.choice(self.config["channels_options"])
                    layer["kernel"] = random.choice(self.config["kernel_options"])
                elif layer_type == "pool":
                    layer["kernel"] = random.choice([2, 3])
                    layer["stride"] = random.choice([2, 3])
                elif layer_type == "fc":
                    layer["units"] = random.choice(self.config["fc_units_options"])
                elif layer_type == "detection_head":
                    layer["num_anchors"] = random.choice([3, 4, 5])

        return structure

    def generate(self, mode: str = "genetic") -> Dict[str, Any]:
        """
        Unified entry point for network generation.

        This is the main method called by CLI.
        It builds model_info, and fills random weights.

        Args:
            mode: "random" or "genetic"

        Returns:
            Complete model_info dict with random weights.
        """
        task_type = self.config.get("task_type", "classification")
        info(f"Generating {task_type} network (mode={mode})")

        if mode == "random":
            num_samples = self.config.get("num_samples", 100)
            structure = self.generate_random(num_samples)
        elif mode == "genetic":
            structure = self.generate_evolved()
        else:
            fatal_error(f"Unknown generation mode: {mode}")

        # Convert structure to model_info
        model_info = self._structure_to_model_info(structure)

        info(f"Generated network: {len(structure['layers'])} layers")

        return model_info
