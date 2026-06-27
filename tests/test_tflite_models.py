#!/usr/bin/env python3
"""
Download tflite models from tflite-micro repository and test them with TinyMLC
"""

import urllib.request
import urllib.error
from pathlib import Path

from TinyMLC.converter.parser_litert import parse_model_tflite

# Models to download from tflite-micro
MODELS = [
    # hello_world models
    (
        "hello_world_float.tflite",
        "https://raw.githubusercontent.com/tensorflow/tflite-micro/main/"
        "tensorflow/lite/micro/examples/hello_world/"
        "models/hello_world_float.tflite"
    ),
    (
        "hello_world_int8.tflite",
        "https://raw.githubusercontent.com/tensorflow/tflite-micro/main/"
        "tensorflow/lite/micro/examples/hello_world/"
        "models/hello_world_int8.tflite"
    ),
    # mnist_lstm models
    (
        "trained_lstm_float.tflite",
        "https://raw.githubusercontent.com/tensorflow/tflite-micro/main/"
        "tensorflow/lite/micro/examples/mnist_lstm/trained_lstm.tflite"
    ),
    # micro_speech models
    (
        "micro_speech_quantized.tflite",
        "https://raw.githubusercontent.com/tensorflow/tflite-micro/main/"
        "tensorflow/lite/micro/examples/micro_speech/"
        "models/micro_speech_quantized.tflite"
    ),
    # memory_footprint models
    (
        "simple_add_model.tflite",
        "https://raw.githubusercontent.com/tensorflow/tflite-micro/main/"
        "tensorflow/lite/micro/examples/memory_footprint/"
        "models/simple_add_model.tflite"
    ),
    # dtln models
    (
        "dtln_noise_suppression.tflite",
        "https://raw.githubusercontent.com/tensorflow/tflite-micro/main/"
        "tensorflow/lite/micro/examples/dtln/dtln_noise_suppression.tflite"
    ),
]

# Integration test models (seanet)
SEANET_MODELS = [
    # ADD models
    ("seanet_add0.tflite", "add0.tflite"),
    ("seanet_add1.tflite", "add1.tflite"),
    ("seanet_add2.tflite", "add2.tflite"),
    # CONV models
    ("seanet_conv0.tflite", "conv0.tflite"),
    ("seanet_conv1.tflite", "conv1.tflite"),
    # FULLY_CONNECTED models
    ("seanet_fully_connected0.tflite", "fully_connected0.tflite"),
    ("seanet_fully_connected1.tflite", "fully_connected1.tflite"),
]


def download_model(name, url, output_dir):
    """Download a model from URL"""
    output_path = output_dir / name
    if output_path.exists():
        print(f"  [SKIP] {name} already exists")
        return True

    print(f"  [DOWNLOAD] {name}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, output_path)
        print("OK")
        return True
    except urllib.error.URLError as e:
        print(f"FAILED: {e}")
        return False


def test_model(model_path):
    """Test parsing a tflite model"""
    print(f"  [TEST] {model_path.name}...", end=" ", flush=True)
    try:
        info = parse_model_tflite(str(model_path))
        ops = [op["op_name"] for op in info["ops"]]
        print(f"OK ({len(ops)} operators)")
        return True, ops
    except Exception as e:
        print(f"FAILED: {e}")
        return False, []


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "test_models"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Downloading tflite models from tflite-micro repository")
    print("=" * 60)

    downloaded = []
    for name, url in MODELS:
        if download_model(name, url, output_dir):
            downloaded.append(name)

    print("\n" + "=" * 60)
    print("Downloading seanet integration test models")
    print("=" * 60)

    base_url = (
        "https://raw.githubusercontent.com/tensorflow/tflite-micro/main/"
        "tensorflow/lite/micro/integration_tests/seanet/"
    )

    for output_name, model_name in SEANET_MODELS:
        url = base_url + model_name
        if download_model(output_name, url, output_dir):
            downloaded.append(output_name)

    print("\n" + "=" * 60)
    print("Testing all downloaded tflite models")
    print("=" * 60)

    tflite_files = list(output_dir.glob("*.tflite"))
    print(f"\nFound {len(tflite_files)} tflite files\n")

    success_count = 0
    fail_count = 0
    summary = {}

    for model_path in sorted(tflite_files):
        success, ops = test_model(model_path)
        if success:
            success_count += 1
            summary[model_path.name] = ops
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total: {len(tflite_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")

    print("\nModels tested successfully:")
    for name, ops in sorted(summary.items()):
        print(f"  {name}: {ops}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
