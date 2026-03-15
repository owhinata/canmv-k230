"""Run MLPerf Tiny IC benchmark on K230 DUT using the upstream runner.

This script:
1. Prepares the IC evaluation dataset (CIFAR-10 perf samples as .bin files)
2. Launches the DUT on K230 via serial
3. Runs the upstream runner's Script engine for accuracy and performance
4. Saves results to the submission directory

Usage:
    .venv/bin/python apps/mlperf_tiny/scripts/run_benchmark.py
    .venv/bin/python apps/mlperf_tiny/scripts/run_benchmark.py --mode p  # performance
    .venv/bin/python apps/mlperf_tiny/scripts/run_benchmark.py --mode a  # accuracy

Prerequisites:
    pip install -r apps/mlperf_tiny/scripts/requirements.txt
    DUT deployed: cmake --build build/mlperf_tiny --target deploy
"""

import argparse
import csv
import os
import pickle
import struct
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
RUNNER_DIR = os.path.join(REPO_ROOT, "mlperf_tiny", "benchmark", "runner")
TRAINING_DIR = os.path.join(
    REPO_ROOT, "mlperf_tiny", "benchmark", "training", "image_classification"
)
EVAL_DIR = os.path.join(
    REPO_ROOT, "mlperf_tiny", "benchmark", "evaluation", "datasets", "ic01"
)
RESULTS_DIR = os.path.join(
    SCRIPT_DIR, "..", "submission", "measurements", "k230_kpu_rtsmart", "ic"
)


def prepare_dataset():
    """Generate IC evaluation dataset (.bin files) from CIFAR-10."""
    idxs_path = os.path.join(TRAINING_DIR, "perf_samples_idxs.npy")
    if not os.path.exists(idxs_path):
        print(f"ERROR: {idxs_path} not found")
        return False

    # Check if dataset already exists
    labels_path = os.path.join(EVAL_DIR, "y_labels.csv")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            entries = list(csv.reader(f))
        bin_count = sum(
            1 for e in entries
            if os.path.exists(os.path.join(EVAL_DIR, e[0]))
        )
        if bin_count == len(entries):
            print(f"  Dataset already prepared ({bin_count} files)")
            return True

    print("  Generating IC evaluation dataset from CIFAR-10...")

    # Load CIFAR-10 test data
    import glob
    keras_dir = os.path.expanduser("~/.keras/datasets")
    candidates = glob.glob(
        os.path.join(keras_dir, "**/test_batch"), recursive=True
    )
    if not candidates:
        print("  ERROR: CIFAR-10 not cached. Run golden_test.py first.")
        return False

    data_dir = os.path.dirname(candidates[0])

    # Load all test batches
    test_data = []
    test_labels = []
    test_filenames = []
    with open(os.path.join(data_dir, "test_batch"), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    # CIFAR-10 data is stored as (N, 3072) in CHW order: 1024R, 1024G, 1024B
    test_data = d[b"data"]  # (10000, 3072) uint8
    test_labels_raw = np.array(d[b"labels"])  # (10000,)
    test_filenames = d[b"filenames"]  # list of bytes

    # One-hot encode labels for compatibility with perf_samples_loader
    test_labels = np.zeros((len(test_labels_raw), 10))
    for i, l in enumerate(test_labels_raw):
        test_labels[i, l] = 1

    # Load perf sample indices
    idxs = np.load(idxs_path)
    print(f"  Generating {len(idxs)} samples...")

    os.makedirs(EVAL_DIR, exist_ok=True)

    with open(labels_path, "w") as label_file:
        for i in idxs:
            filename = test_filenames[i].decode("UTF-8")
            bin_name = filename[:-3] + "bin" if filename.endswith("png") else filename
            label = int(np.argmax(test_labels[i]))

            label_file.write(f"{bin_name},10,{label}\n")

            # The evaluation format is U8C3 RGB where [0]=upper-left-corner
            # CIFAR-10 raw is 1024R+1024G+1024B (CHW), need to convert to
            # HWC interleaved: pixel[0]=(R,G,B), pixel[1]=(R,G,B), ...
            chw = test_data[i].reshape(3, 32, 32)  # (C, H, W)
            hwc = chw.transpose(1, 2, 0).flatten()  # (H, W, C) -> flat

            bin_path = os.path.join(EVAL_DIR, bin_name)
            with open(bin_path, "wb") as f:
                f.write(struct.pack(f"{len(hwc)}B", *hwc))

    print(f"  Generated {len(idxs)} .bin files + y_labels.csv")
    return True


def run_benchmark(args):
    """Run the benchmark using upstream runner components."""
    # Prepare dataset
    print("[1/4] Preparing dataset...")
    if not prepare_dataset():
        return 1

    # Add runner and scripts dirs to path
    sys.path.insert(0, RUNNER_DIR)
    sys.path.insert(0, SCRIPT_DIR)
    from k230_serial_device import K230SerialDevice
    from device_under_test import DUT
    from script import Script
    from datasets import DataSet

    # Load test config
    import yaml
    test_script_path = os.path.join(RUNNER_DIR, f"tests_{args.mode_name}.yaml")
    with open(test_script_path) as f:
        test_scripts = yaml.load(f, Loader=yaml.SafeLoader)

    ic_config = test_scripts.get("ic01")
    if not ic_config:
        print(f"ERROR: ic01 not found in {test_script_path}")
        return 1

    print(f"[2/4] Test config: {ic_config}")

    # Connect to K230
    print(f"[3/4] Connecting to {args.port}...")
    port = K230SerialDevice(args.port, args.baud, echo=args.echo)

    with port:
        # Launch DUT
        print("  Launching DUT on K230...")
        port.write_line("\x03")  # Kill any running process
        time.sleep(0.5)
        while not port._message_queue.empty():
            port._message_queue.get_nowait()
        port.write("\r")
        time.sleep(0.5)
        while not port._message_queue.empty():
            port._message_queue.get_nowait()

        dut_cmd = "/sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/model.kmodel"
        port.write(f"{dut_cmd}\r")
        # Wait for m-ready
        lines = []
        t0 = time.time()
        while time.time() - t0 < 15:
            try:
                line = port._message_queue.get(timeout=1)
                lines.append(line)
                if "m-ready" in line:
                    break
            except Exception:
                pass
        if not any("m-ready" in l for l in lines):
            print(f"ERROR: DUT did not reach m-ready: {lines}")
            return 1
        print("  DUT ready")

        # Create DUT wrapper
        dut = DUT(port, baud_rate=args.baud)
        dut._name = "k230"
        dut._model = "ic01"
        dut._profile = "ULPMark for tinyML Firmware V0.0.1"

        # Load dataset
        dataset = DataSet(EVAL_DIR, ic_config["truth_file"])
        print(f"  Dataset: {dataset.get_num_files()} files")

        # Run script
        print(f"\n[4/4] Running {args.mode_name} benchmark...")
        script = Script(ic_config)
        result = script.run(None, dut, dataset, args.mode)

        # Save results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        import json
        results_path = os.path.join(RESULTS_DIR, "results.json")
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")

        # Exit DUT
        port.write_line("exit")
        time.sleep(1)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run MLPerf Tiny IC benchmark on K230"
    )
    parser.add_argument(
        "--mode", choices=["a", "p"], default="a",
        help="Benchmark mode: a=accuracy, p=performance (default: a)",
    )
    parser.add_argument(
        "--port", default="/dev/ttyACM1",
        help="Serial port (default: /dev/ttyACM1)",
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
    )
    parser.add_argument(
        "--echo", action="store_true",
        help="Echo serial communication for debugging",
    )
    args = parser.parse_args()

    mode_names = {"a": "accuracy", "p": "performance"}
    args.mode_name = mode_names[args.mode]

    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
