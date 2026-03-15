"""Golden inference comparison for MLPerf Tiny Image Classification on K230.

Loads the reference TFLite model (pretrainedResnet.tflite, float32) and
CIFAR-10 test samples, runs inference on both the TFLite model and a K230
DUT connected via serial, then compares the results.

The TFLite model input is float32 [1, 32, 32, 3] NHWC, range [0, 1].
The DUT receives uint8 data via the MLPerf Tiny UART protocol and internally
converts uint8 [0,255] -> float32 [0,1] by dividing by 255.

Both paths receive equivalent input: TFLite gets float32 = uint8 / 255.0,
and the DUT gets the same uint8 bytes.

Usage:
    .venv/bin/python apps/mlperf_tiny/scripts/golden_test.py
    .venv/bin/python apps/mlperf_tiny/scripts/golden_test.py -n 50 --port /dev/ttyACM1

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
TFLITE_PATH = os.path.join(
    REPO_ROOT,
    "mlperf_tiny/benchmark/training/image_classification/"
    "trained_models/pretrainedResnet.tflite",
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

INPUT_SIZE = 32 * 32 * 3  # 3072 bytes
NUM_CLASSES = 10
DUT_MAX_HEX_BYTES = 31  # max bytes per "db" command (no power manager)


# ---------------------------------------------------------------------------
# TFLite inference
# ---------------------------------------------------------------------------

def load_tflite_model(model_path):
    """Load TFLite model and return an interpreter ready for inference."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def tflite_infer(interpreter, image_uint8):
    """Run TFLite inference on a single uint8 image.

    Args:
        interpreter: TFLite Interpreter instance.
        image_uint8: numpy array of shape (32, 32, 3), dtype uint8.

    Returns:
        numpy array of shape (10,), the output probabilities/logits.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # The reference model expects raw uint8 values cast to float32 [0,255]
    image_float = image_uint8.astype(np.float32)
    input_data = np.expand_dims(image_float, axis=0)  # (1, 32, 32, 3)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    return output[0]  # shape (10,)


# ---------------------------------------------------------------------------
# CIFAR-10 data loading
# ---------------------------------------------------------------------------

def load_cifar10_test():
    """Load CIFAR-10 test set. Returns (images, labels).

    images: (10000, 32, 32, 3) uint8
    labels: (10000,) int
    """
    import glob
    import pickle

    # Try Keras cache first (handles various directory layouts)
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
        return images, labels

    # Fallback: download via Keras
    from tensorflow.keras.datasets import cifar10
    (_, _), (x_test, y_test) = cifar10.load_data()
    return x_test, y_test.flatten()


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
        """Read lines until 'm-ready' is seen or timeout. Returns list of lines."""
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
        """Load uint8 data into DUT buffer via 'db load' + 'db HH...' commands.

        Args:
            data_uint8: flat numpy array or bytes of uint8 values.
        """
        data = bytes(data_uint8)
        self.send_command(f"db load {len(data)}")

        # EE_CMD_SIZE is 80 chars.  "db " = 3 chars, so max hex payload
        # is 77 chars → 38 bytes.  Each command gets m-ready response.
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


def parse_dut_response(lines):
    """Parse DUT response lines for results and timestamps.

    Returns:
        results: list of 10 floats, or None if not found.
        cycles_start: int or None (rdcycle before inference).
        cycles_end: int or None (rdcycle after inference).
    """
    results = None
    timestamps = []

    for line in lines:
        # m-results-[v0,v1,...,v9]
        match = re.search(r"m-results-\[([^\]]+)\]", line)
        if match:
            try:
                results = [float(x) for x in match.group(1).split(",")]
            except ValueError:
                pass

        # m-lap-us-XXXX (actually rdcycle counts on K230)
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
    # Load TFLite model
    print(f"Loading TFLite model: {args.tflite}")
    if not os.path.exists(args.tflite):
        print(f"ERROR: TFLite model not found: {args.tflite}")
        print("Run: git submodule update --init mlperf_tiny")
        return 1
    interpreter = load_tflite_model(args.tflite)

    # Load CIFAR-10 test data
    print("Loading CIFAR-10 test set...")
    x_test, y_test = load_cifar10_test()
    print(f"  Loaded {len(x_test)} test samples")

    n = min(args.n, len(x_test))
    print(f"  Using first {n} samples")

    # Connect to DUT and launch
    print(f"Connecting to DUT: {args.port} @ {args.baud}")
    dut = K230DUT(args.port, args.baud, timeout=args.timeout)

    dut_cmd = "/sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/model.kmodel"
    try:
        print(f"Launching DUT: {dut_cmd}")
        dut.launch_dut(dut_cmd)

        # Tracking
        tflite_correct = 0
        dut_correct = 0
        agree_count = 0
        dut_latencies = []

        print(f"\n{'idx':>5}  {'label':>10}  {'tflite':>10}  {'dut':>10}  "
              f"{'match':>5}  {'cycles':>12}")
        print("-" * 65)

        for i in range(n):
            image_uint8 = x_test[i]  # (32, 32, 3) uint8
            label = int(y_test[i])

            # --- TFLite inference ---
            tflite_out = tflite_infer(interpreter, image_uint8)
            tflite_class = int(np.argmax(tflite_out))

            # --- DUT inference ---
            # Flatten to C-contiguous uint8 bytes (NHWC, row-major)
            data_flat = image_uint8.flatten().astype(np.uint8)
            dut.load_data(data_flat)
            dut_lines = dut.infer(n=1, warmups=0, timeout=args.timeout)

            dut_results, cyc_start, cyc_end = parse_dut_response(dut_lines)

            if dut_results is not None and len(dut_results) == NUM_CLASSES:
                dut_class = int(np.argmax(dut_results))
            else:
                dut_class = -1
                print(f"  WARNING: sample {i} - failed to parse DUT results")
                if dut_lines:
                    for line in dut_lines:
                        print(f"    DUT> {line}")

            # --- Latency ---
            cycles = None
            if cyc_start is not None and cyc_end is not None:
                cycles = cyc_end - cyc_start
                dut_latencies.append(cycles)

            # --- Comparison ---
            tflite_ok = tflite_class == label
            dut_ok = dut_class == label
            agreed = tflite_class == dut_class

            if tflite_ok:
                tflite_correct += 1
            if dut_ok:
                dut_correct += 1
            if agreed:
                agree_count += 1

            cycles_str = f"{cycles:>12,}" if cycles is not None else "         N/A"
            match_str = "Y" if agreed else "N"

            print(f"{i:>5}  {CIFAR10_CLASSES[label]:>10}  "
                  f"{CIFAR10_CLASSES[tflite_class]:>10}  "
                  f"{CIFAR10_CLASSES[dut_class] if dut_class >= 0 else 'ERR':>10}  "
                  f"{match_str:>5}  {cycles_str}")

        # --- Summary ---
        print("\n" + "=" * 65)
        print("Summary")
        print("=" * 65)
        print(f"  Samples tested   : {n}")
        print(f"  TFLite accuracy  : {tflite_correct}/{n} "
              f"({100.0 * tflite_correct / n:.1f}%)")
        print(f"  DUT accuracy     : {dut_correct}/{n} "
              f"({100.0 * dut_correct / n:.1f}%)")
        print(f"  Agreement rate   : {agree_count}/{n} "
              f"({100.0 * agree_count / n:.1f}%)")

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
        description="Golden inference comparison: TFLite vs K230 DUT "
                    "(MLPerf Tiny Image Classification)"
    )
    parser.add_argument(
        "-n", type=int, default=100,
        help="Number of CIFAR-10 test samples to evaluate (default: 100)",
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
        "--tflite", default=TFLITE_PATH,
        help="Path to float32 TFLite model",
    )
    args = parser.parse_args()

    return run_golden_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
