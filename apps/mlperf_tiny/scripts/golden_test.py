"""Golden inference comparison for MLPerf Tiny benchmarks on K230.

Loads the reference TFLite model and test samples, runs inference on both the
TFLite model and a K230 DUT connected via serial, then compares the results.

Supports all benchmarks: IC (CIFAR-10), VWW, KWS, AD.

Usage:
    .venv/bin/python golden_test.py --benchmark ic01
    .venv/bin/python golden_test.py --benchmark vww01 -n 50
    .venv/bin/python golden_test.py --benchmark ad01 --port /dev/ttyACM1

Prerequisites:
    pip install -r requirements.txt
"""

import argparse
import os
import re
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import serial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TRAINING_DIR = os.path.join(REPO_ROOT, "mlperf_tiny", "benchmark", "training")

DUT_MAX_HEX_BYTES = 31

# Benchmark definitions
BENCHMARK_CONFIG = {
    "ic01": {
        "tflite": os.path.join(
            TRAINING_DIR,
            "image_classification/trained_models/pretrainedResnet.tflite",
        ),
        "input_shape": (32, 32, 3),
        "output_elements": 10,
        "task": "classification",
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
    },
    "vww01": {
        "tflite": os.path.join(
            TRAINING_DIR,
            "visual_wake_words/trained_models/vww_96_float.tflite",
        ),
        "input_shape": (96, 96, 3),
        "output_elements": 2,
        "task": "classification",
        "class_names": ["not_person", "person"],
    },
    "kws01": {
        "tflite": os.path.join(
            TRAINING_DIR,
            "keyword_spotting/trained_models/kws_ref_model_float32.tflite",
        ),
        "input_shape": (49, 10, 1),
        "output_elements": 12,
        "task": "classification",
        "class_names": [
            "silence", "unknown", "yes", "no", "up", "down",
            "left", "right", "on", "off", "stop", "go",
        ],
    },
    "ad01": {
        "tflite": os.path.join(
            TRAINING_DIR,
            "anomaly_detection/trained_models/ad01_fp32.tflite",
        ),
        "input_shape": (640,),
        "output_elements": 640,
        "task": "regression",
        "class_names": None,
    },
}


# ---------------------------------------------------------------------------
# TFLite inference
# ---------------------------------------------------------------------------

def load_tflite_model(model_path):
    """Load TFLite model and return an interpreter ready for inference."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def tflite_infer(interpreter, data):
    """Run TFLite inference on a single input.

    Args:
        interpreter: TFLite Interpreter instance.
        data: numpy array (uint8 for classification, float32 for AD).

    Returns:
        numpy array of output values.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.expand_dims(data.astype(np.float32), axis=0)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    return output[0]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_ic01(n):
    """Load CIFAR-10 test set. Returns (data, labels)."""
    import glob
    import pickle

    keras_dir = os.path.expanduser("~/.keras/datasets")
    candidates = glob.glob(
        os.path.join(keras_dir, "**/test_batch"), recursive=True
    )
    if candidates:
        data_dir = os.path.dirname(candidates[0])
        print(f"  Found cached CIFAR-10: {data_dir}")
        with open(os.path.join(data_dir, "test_batch"), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        images = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(d[b"labels"])
        return images[:n], labels[:n]

    from tensorflow.keras.datasets import cifar10
    (_, _), (x_test, y_test) = cifar10.load_data()
    return x_test[:n], y_test.flatten()[:n]


def load_dataset_from_eval_dir(benchmark, n):
    """Load dataset from evaluation directory (.bin files + y_labels.csv).

    Works for any benchmark that has pre-populated evaluation data.
    For AD: .bin files are full spectrograms; extracts first sliding window.
    """
    eval_dir = os.path.join(
        REPO_ROOT, "mlperf_tiny", "benchmark", "evaluation", "datasets", benchmark
    )
    labels_path = os.path.join(eval_dir, "y_labels.csv")
    if not os.path.exists(labels_path):
        print(f"  WARNING: {labels_path} not found, using random data")
        cfg = BENCHMARK_CONFIG[benchmark]
        shape = cfg["input_shape"]
        data = np.random.randint(0, 256, (n,) + shape, dtype=np.uint8)
        return data, np.zeros(n, dtype=int)

    import csv
    with open(labels_path) as f:
        entries = list(csv.reader(f))

    cfg = BENCHMARK_CONFIG[benchmark]
    shape = cfg["input_shape"]
    is_ad = cfg["task"] == "regression"
    data_list = []
    label_list = []
    for entry in entries[:n]:
        bin_name = entry[0].strip()
        label = int(entry[2].strip())
        bin_path = os.path.join(eval_dir, bin_name)
        if not os.path.exists(bin_path):
            continue

        if is_ad:
            # AD: .bin is full spectrogram (float32), extract first window
            # y_labels format: filename,2,label,window_width_bytes,stride_bytes
            window_width = int(entry[3].strip())  # bytes (e.g. 2560)
            n_floats = window_width // 4  # 640
            raw = np.fromfile(bin_path, dtype=np.float32)
            window = raw[:n_floats]
            data_list.append(window)
        else:
            raw = np.fromfile(bin_path, dtype=np.uint8)
            data_list.append(raw.reshape(shape))
        label_list.append(label)

    if not data_list:
        print(f"  WARNING: No .bin files found in {eval_dir}, using random data")
        data = np.random.randint(0, 256, (n,) + shape, dtype=np.uint8)
        return data, np.zeros(n, dtype=int)

    print(f"  Loaded {len(data_list)} samples from {eval_dir}")
    return np.array(data_list), np.array(label_list)


def load_dataset(benchmark, n):
    """Load dataset for the specified benchmark."""
    if benchmark == "ic01":
        return load_dataset_ic01(n)
    return load_dataset_from_eval_dir(benchmark, n)


# ---------------------------------------------------------------------------
# K230 DUT serial communication
# ---------------------------------------------------------------------------

class K230DUT:
    """Communicates with K230 DUT via the MLPerf Tiny UART protocol."""

    def __init__(self, port, baud=115200, timeout=5.0):
        self._ser = serial.Serial(port, baud, timeout=0.1)
        self._timeout = timeout

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _write(self, cmd):
        """Send a command terminated with % and CR."""
        self._ser.write(f"{cmd}%\r".encode())

    def _read_until_ready(self, timeout=None):
        """Read lines until 'm-ready' is seen or timeout."""
        if timeout is None:
            timeout = self._timeout
        lines = []
        t0 = time.time()
        buf = ""
        while time.time() - t0 < timeout:
            raw = self._ser.read(256)
            if not raw:
                continue
            try:
                buf += raw.decode(errors="replace")
            except Exception:
                continue
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip("\r\x00 ")
                if not line:
                    continue
                lines.append(line)
                if "m-ready" in line:
                    return lines
        return lines

    def flush(self):
        """Flush DUT buffers by sending an empty command."""
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        self._write("")
        time.sleep(0.2)
        self._ser.reset_input_buffer()

    def send_command(self, cmd, timeout=None):
        """Send command and collect response lines until m-ready."""
        self._write(cmd)
        return self._read_until_ready(timeout=timeout)

    def launch_dut(self, cmd):
        """Launch DUT binary on msh. Sends Ctrl+C first to kill any prior."""
        self._ser.reset_input_buffer()
        self._ser.write(b"\x03")
        time.sleep(0.5)
        self._ser.read(self._ser.in_waiting or 1)
        self._ser.write(b"\r")
        time.sleep(0.5)
        self._ser.read(self._ser.in_waiting or 1)
        self._ser.write(f"{cmd}\r".encode())
        lines = self._read_until_ready(timeout=10.0)
        if not any("m-ready" in l for l in lines):
            raise RuntimeError(f"DUT did not reach m-ready: {lines}")
        return lines

    def load_data(self, data_uint8):
        """Load uint8 data into DUT buffer via 'db load' + 'db HH...' commands."""
        data = bytes(data_uint8)
        self.send_command(f"db load {len(data)}")

        chunk_bytes = 38
        i = 0
        while i < len(data):
            chunk = data[i:i + chunk_bytes]
            hex_str = "".join(f"{b:02x}" for b in chunk)
            self.send_command(f"db {hex_str}")
            i += chunk_bytes

    def infer(self, n=1, warmups=0, timeout=30.0):
        """Run inference on DUT. Returns response lines."""
        return self.send_command(f"infer {n} {warmups}", timeout=timeout)


def parse_dut_response(lines, output_elements):
    """Parse DUT response lines for results and timestamps.

    Returns:
        results: list of floats, or None if not found.
        cycles_start: int or None.
        cycles_end: int or None.
    """
    results = None
    timestamps = []

    for line in lines:
        match = re.search(r"m-results-\[([^\]]+)\]", line)
        if match:
            try:
                results = [float(x) for x in match.group(1).split(",")]
            except ValueError:
                pass

        match = re.search(r"m-lap-us-(\d+)", line)
        if match:
            timestamps.append(int(match.group(1)))

    cycles_start = timestamps[0] if len(timestamps) >= 1 else None
    cycles_end = timestamps[1] if len(timestamps) >= 2 else None

    return results, cycles_start, cycles_end


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------

def run_golden_test(args):
    """Run golden comparison between TFLite and K230 DUT."""
    benchmark = args.benchmark
    cfg = BENCHMARK_CONFIG[benchmark]
    tflite_path = args.tflite or cfg["tflite"]
    output_elements = cfg["output_elements"]
    is_classification = cfg["task"] == "classification"
    class_names = cfg["class_names"]

    # Load TFLite model
    print(f"Loading TFLite model: {tflite_path}")
    if not os.path.exists(tflite_path):
        print(f"ERROR: TFLite model not found: {tflite_path}")
        print("Run: git submodule update --init mlperf_tiny")
        return 1
    interpreter = load_tflite_model(tflite_path)

    # Load test data
    print(f"Loading test data for {benchmark}...")
    x_test, y_test = load_dataset(benchmark, args.n)
    n = len(x_test)
    print(f"  Using {n} samples")

    # Connect to DUT and launch
    print(f"Connecting to DUT: {args.port} @ {args.baud}")
    dut = K230DUT(args.port, args.baud, timeout=args.timeout)

    kmodel_path = f"/sharefs/mlperf_tiny/{benchmark}.kmodel"
    dut_cmd = f"/sharefs/mlperf_tiny/mlperf_tiny {kmodel_path}"
    try:
        print(f"Launching DUT: {dut_cmd}")
        dut.launch_dut(dut_cmd)

        tflite_correct = 0
        dut_correct = 0
        agree_count = 0
        dut_latencies = []

        if is_classification:
            print(f"\n{'idx':>5}  {'label':>10}  {'tflite':>10}  {'dut':>10}  "
                  f"{'match':>5}  {'cycles':>12}")
            print("-" * 65)
        else:
            print(f"\n{'idx':>5}  {'mse':>12}  {'cycles':>12}")
            print("-" * 35)

        for i in range(n):
            sample = x_test[i]
            label = int(y_test[i])

            # --- TFLite inference ---
            tflite_out = tflite_infer(interpreter, sample)

            # --- DUT inference ---
            if is_classification:
                # Quantized input: send as uint8 bytes
                data_flat = sample.flatten().astype(np.uint8)
            else:
                # Float32 input (AD): send raw float32 bytes
                data_flat = sample.flatten().astype(np.float32).tobytes()
            dut.load_data(data_flat)
            dut_lines = dut.infer(n=1, warmups=0, timeout=args.timeout)

            dut_results, cyc_start, cyc_end = parse_dut_response(
                dut_lines, output_elements
            )

            # --- Latency ---
            cycles = None
            if cyc_start is not None and cyc_end is not None:
                cycles = cyc_end - cyc_start
                dut_latencies.append(cycles)

            cycles_str = f"{cycles:>12,}" if cycles is not None else "         N/A"

            if is_classification:
                tflite_class = int(np.argmax(tflite_out))

                if dut_results is not None and len(dut_results) == output_elements:
                    dut_class = int(np.argmax(dut_results))
                else:
                    dut_class = -1
                    print(f"  WARNING: sample {i} - failed to parse DUT results")
                    if dut_lines:
                        for line in dut_lines:
                            print(f"    DUT> {line}")

                tflite_ok = tflite_class == label
                dut_ok = dut_class == label
                agreed = tflite_class == dut_class

                if tflite_ok:
                    tflite_correct += 1
                if dut_ok:
                    dut_correct += 1
                if agreed:
                    agree_count += 1

                match_str = "Y" if agreed else "N"
                tflite_name = class_names[tflite_class] if class_names else str(tflite_class)
                dut_name = class_names[dut_class] if class_names and dut_class >= 0 else "ERR"
                label_name = class_names[label] if class_names else str(label)

                print(f"{i:>5}  {label_name:>10}  {tflite_name:>10}  "
                      f"{dut_name:>10}  {match_str:>5}  {cycles_str}")
            else:
                # Regression (AD): compare MSE
                if dut_results is not None and len(dut_results) == output_elements:
                    dut_arr = np.array(dut_results)
                    mse = float(np.mean((tflite_out - dut_arr) ** 2))
                    agree_count += 1
                else:
                    mse = float("nan")
                    print(f"  WARNING: sample {i} - failed to parse DUT results")

                print(f"{i:>5}  {mse:>12.4f}  {cycles_str}")

        # --- Summary ---
        print("\n" + "=" * 65)
        print(f"Summary ({benchmark})")
        print("=" * 65)
        print(f"  Samples tested   : {n}")

        if is_classification:
            print(f"  TFLite accuracy  : {tflite_correct}/{n} "
                  f"({100.0 * tflite_correct / n:.1f}%)")
            print(f"  DUT accuracy     : {dut_correct}/{n} "
                  f"({100.0 * dut_correct / n:.1f}%)")
            print(f"  Agreement rate   : {agree_count}/{n} "
                  f"({100.0 * agree_count / n:.1f}%)")
        else:
            print(f"  Valid results    : {agree_count}/{n}")

        if dut_latencies:
            lat = np.array(dut_latencies, dtype=np.int64)
            print(f"\n  DUT latency (rdcycle counts):")
            print(f"    min    : {lat.min():>12,}")
            print(f"    max    : {lat.max():>12,}")
            print(f"    mean   : {lat.mean():>12,.0f}")
            print(f"    median : {int(np.median(lat)):>12,}")

    finally:
        print("\nStopping DUT...")
        dut.send_command("exit", timeout=3.0)
        dut.close()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Golden inference comparison: TFLite vs K230 DUT"
    )
    parser.add_argument(
        "--benchmark", required=True,
        choices=list(BENCHMARK_CONFIG.keys()),
        help="Benchmark to test",
    )
    parser.add_argument(
        "-n", type=int, default=100,
        help="Number of test samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--port", default="/dev/ttyACM1",
        help="Serial port for K230 DUT (default: /dev/ttyACM1)",
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Baud rate (default: 115200)",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0,
        help="Per-command serial timeout in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--tflite", default=None,
        help="Path to float32 TFLite model (default: benchmark-specific)",
    )
    args = parser.parse_args()

    return run_golden_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
