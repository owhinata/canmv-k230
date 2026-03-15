"""Run MLPerf Tiny benchmark on K230 DUT using the upstream runner.

This script:
1. Prepares the evaluation dataset for the specified benchmark
2. Launches the DUT on K230 via serial
3. Runs the upstream runner's Script engine for accuracy and performance
4. Saves results to the submission directory

Usage:
    .venv/bin/python run_benchmark.py --benchmark ic01
    .venv/bin/python run_benchmark.py --benchmark vww01 --mode p
    .venv/bin/python run_benchmark.py --benchmark kws01 --mode a

Prerequisites:
    pip install -r requirements.txt
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
TRAINING_DIR = os.path.join(REPO_ROOT, "mlperf_tiny", "benchmark", "training")
EVAL_BASE_DIR = os.path.join(
    REPO_ROOT, "mlperf_tiny", "benchmark", "evaluation", "datasets"
)
SUBMISSION_BASE_DIR = os.path.join(
    SCRIPT_DIR, "..", "submission", "measurements", "k230_kpu_rtsmart"
)

# Benchmark-specific configuration
BENCHMARK_CONFIG = {
    "ic01": {
        "name": "image_classification",
        "training_subdir": "image_classification",
        "dataset_prep": "cifar10",
    },
    "vww01": {
        "name": "visual_wake_words",
        "training_subdir": "visual_wake_words",
        "dataset_prep": "vww",
    },
    "kws01": {
        "name": "keyword_spotting",
        "training_subdir": "keyword_spotting",
        "dataset_prep": "kws",
    },
    "ad01": {
        "name": "anomaly_detection",
        "training_subdir": "anomaly_detection",
        "dataset_prep": "ad",
    },
}


def prepare_dataset_ic01(eval_dir):
    """Generate IC evaluation dataset (.bin files) from CIFAR-10."""
    training_dir = os.path.join(TRAINING_DIR, "image_classification")
    idxs_path = os.path.join(training_dir, "perf_samples_idxs.npy")
    if not os.path.exists(idxs_path):
        print(f"ERROR: {idxs_path} not found")
        return False

    labels_path = os.path.join(eval_dir, "y_labels.csv")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            entries = list(csv.reader(f))
        bin_count = sum(
            1 for e in entries
            if os.path.exists(os.path.join(eval_dir, e[0]))
        )
        if bin_count == len(entries):
            print(f"  Dataset already prepared ({bin_count} files)")
            return True

    print("  Generating IC evaluation dataset from CIFAR-10...")

    import glob
    keras_dir = os.path.expanduser("~/.keras/datasets")
    candidates = glob.glob(
        os.path.join(keras_dir, "**/test_batch"), recursive=True
    )
    if not candidates:
        print("  ERROR: CIFAR-10 not cached. Run golden_test.py first.")
        return False

    data_dir = os.path.dirname(candidates[0])

    with open(os.path.join(data_dir, "test_batch"), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    test_data = d[b"data"]
    test_labels_raw = np.array(d[b"labels"])
    test_filenames = d[b"filenames"]

    test_labels = np.zeros((len(test_labels_raw), 10))
    for i, l in enumerate(test_labels_raw):
        test_labels[i, l] = 1

    idxs = np.load(idxs_path)
    print(f"  Generating {len(idxs)} samples...")

    os.makedirs(eval_dir, exist_ok=True)

    with open(labels_path, "w") as label_file:
        for i in idxs:
            filename = test_filenames[i].decode("UTF-8")
            bin_name = filename[:-3] + "bin" if filename.endswith("png") else filename
            label = int(np.argmax(test_labels[i]))

            label_file.write(f"{bin_name},10,{label}\n")

            chw = test_data[i].reshape(3, 32, 32)
            hwc = chw.transpose(1, 2, 0).flatten()

            bin_path = os.path.join(eval_dir, bin_name)
            with open(bin_path, "wb") as f:
                f.write(struct.pack(f"{len(hwc)}B", *hwc))

    print(f"  Generated {len(idxs)} .bin files + y_labels.csv")
    return True


def prepare_dataset_vww01(eval_dir):
    """Generate VWW evaluation dataset (.bin files) from COCO2014 96x96.

    Downloads vw_coco2014_96.tar.gz from Silicon Labs, caches in ~/.mlperf/vww/,
    converts JPEG images to raw 96x96x3 uint8 RGB .bin files matching
    the filenames in the upstream y_labels.csv.
    """
    from PIL import Image
    import shutil
    import tarfile
    import urllib.request

    # Check if .bin files already exist alongside y_labels.csv
    labels_path = os.path.join(eval_dir, "y_labels.csv")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            entries = list(csv.reader(f))
        bin_count = sum(
            1 for e in entries
            if os.path.exists(os.path.join(eval_dir, e[0].strip()))
        )
        if bin_count == len(entries):
            print(f"  Dataset already prepared ({bin_count} files)")
            return True

    # Cache directory
    cache_dir = os.path.expanduser("~/.mlperf/vww")
    tar_path = os.path.join(cache_dir, "vw_coco2014_96.tar.gz")
    extract_dir = os.path.join(cache_dir, "vw_coco2014_96")

    # Download if not cached
    if not os.path.exists(extract_dir):
        os.makedirs(cache_dir, exist_ok=True)
        url = "https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz"
        if not os.path.exists(tar_path):
            print(f"  Downloading VWW dataset: {url}")
            urllib.request.urlretrieve(url, tar_path)
            print(f"  Downloaded: {os.path.getsize(tar_path) / 1024 / 1024:.1f} MB")
        print(f"  Extracting to {cache_dir}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(cache_dir)

    # Build index: COCO image ID -> JPEG path
    jpeg_index = {}
    for subdir in ["person", "non_person"]:
        dir_path = os.path.join(extract_dir, subdir)
        if not os.path.isdir(dir_path):
            print(f"  ERROR: Expected directory not found: {dir_path}")
            return False
        for fname in os.listdir(dir_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                # Extract COCO ID from filename (e.g. "COCO_val2014_000000343218.jpg")
                stem = os.path.splitext(fname)[0]
                # Try to extract numeric ID from the end
                parts = stem.split("_")
                coco_id = parts[-1] if parts[-1].isdigit() else stem
                jpeg_index[coco_id] = os.path.join(dir_path, fname)

    # Read y_labels.csv and convert matching images to .bin
    if not os.path.exists(labels_path):
        print(f"  ERROR: y_labels.csv not found: {labels_path}")
        return False

    with open(labels_path) as f:
        entries = list(csv.reader(f))

    os.makedirs(eval_dir, exist_ok=True)
    converted = 0
    missing = 0
    for entry in entries:
        bin_name = entry[0].strip()
        bin_path = os.path.join(eval_dir, bin_name)
        if os.path.exists(bin_path):
            converted += 1
            continue

        # Extract COCO ID from bin filename (e.g. "000000343218.bin" -> "000000343218")
        coco_id = os.path.splitext(bin_name)[0]

        jpeg_path = jpeg_index.get(coco_id)
        if jpeg_path is None:
            # Try without leading zeros
            jpeg_path = jpeg_index.get(coco_id.lstrip("0") or "0")
        if jpeg_path is None:
            missing += 1
            if missing <= 3:
                print(f"  WARNING: No JPEG found for {bin_name}")
            continue

        # Load JPEG, resize to 96x96, save as raw RGB bytes
        img = Image.open(jpeg_path).convert("RGB").resize((96, 96))
        raw = np.array(img, dtype=np.uint8).flatten()
        with open(bin_path, "wb") as f:
            f.write(raw.tobytes())
        converted += 1

    if missing > 3:
        print(f"  WARNING: {missing} images not found in cache")
    print(f"  Prepared {converted}/{len(entries)} .bin files")
    return converted == len(entries)


def prepare_dataset_kws01(eval_dir):
    """Generate KWS evaluation dataset using upstream MFCC pipeline.

    Downloads Speech Commands v2 via tensorflow_datasets, computes MFCC
    features matching the upstream make_bin_files.py pipeline, quantizes
    to int8, and saves as .bin files.  Cached in ~/.mlperf/kws/.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    labels_path = os.path.join(eval_dir, "y_labels.csv")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            entries = list(csv.reader(f))
        bin_count = sum(
            1 for e in entries
            if os.path.exists(os.path.join(eval_dir, e[0].strip()))
        )
        if bin_count == len(entries):
            print(f"  Dataset already prepared ({bin_count} files)")
            return True

    cache_dir = os.path.expanduser("~/.mlperf/kws")
    os.makedirs(cache_dir, exist_ok=True)

    # Use upstream KWS scripts for MFCC parameters
    kws_training_dir = os.path.join(TRAINING_DIR, "keyword_spotting")
    sys.path.insert(0, kws_training_dir)
    import kws_util
    import keras_model as kws_models

    flags, _ = kws_util.parse_command()
    flags.data_dir = cache_dir
    flags.feature_type = "mfcc"

    # Load dataset via tfds (auto-downloads to cache_dir)
    print(f"  Loading Speech Commands v2 (cache: {cache_dir})...")
    ds_test, ds_info = tfds.load(
        "speech_commands", split="test",
        data_dir=cache_dir, with_info=True
    )

    model_settings = kws_models.prepare_model_settings(12, flags)

    # Get quantization params from TFLite model
    tflite_path = os.path.join(
        kws_training_dir, "trained_models", "kws_ref_model.tflite"
    )
    tfl_interp = tf.lite.Interpreter(model_path=tflite_path)
    tfl_interp.allocate_tensors()
    input_details = tfl_interp.get_input_details()
    input_scale, input_zero_point = input_details[0]["quantization"]
    print(f"  Quantization: scale={input_scale}, zp={input_zero_point}")

    # MFCC feature extraction (matching upstream get_dataset.py)
    word_labels = [
        "Down", "Go", "Left", "No", "Off", "On", "Right",
        "Stop", "Up", "Yes", "Silence", "Unknown",
    ]
    sample_rate = model_settings["sample_rate"]
    desired_samples = model_settings["desired_samples"]
    window_size = model_settings["window_size_samples"]
    window_stride = model_settings["window_stride_samples"]
    dct_count = model_settings["dct_coefficient_count"]
    spec_len = model_settings["spectrogram_length"]

    os.makedirs(eval_dir, exist_ok=True)
    count = 0
    num_files = 1000

    print(f"  Generating {num_files} KWS .bin files...")
    file_names = []
    labels = []

    for sample in ds_test.take(num_files):
        audio = tf.cast(sample["audio"], tf.float32)
        label = sample["label"].numpy()

        # Normalize and pad
        audio = audio / tf.reduce_max(tf.abs(audio) + 1e-9)
        audio = tf.pad(audio, [[0, desired_samples - tf.shape(audio)[0]]])
        audio = audio[:desired_samples]

        # MFCC extraction
        stfts = tf.signal.stft(
            audio, frame_length=window_size,
            frame_step=window_stride, window_fn=tf.signal.hann_window
        )
        spectrograms = tf.abs(stfts)
        num_spec_bins = stfts.shape[-1]
        linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
            40, num_spec_bins, sample_rate, 20.0, 4000.0
        )
        mel_spec = tf.tensordot(spectrograms, linear_to_mel, 1)
        log_mel = tf.math.log(mel_spec + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)
        mfccs = mfccs[..., :dct_count]

        # Quantize to int8
        mfcc_q = np.array(
            mfccs.numpy() / input_scale + input_zero_point, dtype=np.int8
        )
        mfcc_q = mfcc_q.reshape(spec_len, dct_count, 1)

        label_str = word_labels[label]
        fname = f"tst_{count:06d}_{label_str}_{label}.bin"
        with open(os.path.join(eval_dir, fname), "wb") as f:
            f.write(mfcc_q.flatten().tobytes())

        file_names.append(fname)
        labels.append(label)
        count += 1

    # Write y_labels.csv
    with open(labels_path, "w") as f:
        for fname, lbl in zip(file_names, labels):
            f.write(f"{fname},12,{lbl}\n")

    print(f"  Generated {count} .bin files + y_labels.csv")
    return True


def prepare_dataset_ad01(eval_dir):
    """Generate AD evaluation dataset using upstream spectrogram pipeline.

    Downloads ToyADMOS/ToyCar from Zenodo, computes mel spectrograms
    matching the upstream common.py pipeline, saves as float32 .bin files.
    Cached in ~/.mlperf/ad/.
    """
    import io
    import urllib.request
    import zipfile

    import librosa

    labels_path = os.path.join(eval_dir, "y_labels.csv")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            entries = list(csv.reader(f))
        bin_count = sum(
            1 for e in entries
            if os.path.exists(os.path.join(eval_dir, e[0].strip()))
        )
        if bin_count == len(entries):
            print(f"  Dataset already prepared ({bin_count} files)")
            return True

    cache_dir = os.path.expanduser("~/.mlperf/ad")
    os.makedirs(cache_dir, exist_ok=True)

    # Download ToyADMOS dev_data_ToyCar
    toycar_dir = os.path.join(cache_dir, "dev_data", "ToyCar")
    if not os.path.exists(toycar_dir):
        for url, name in [
            ("https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1",
             "dev_data_ToyCar.zip"),
            ("https://zenodo.org/record/3727685/files/eval_data_train_ToyCar.zip?download=1",
             "eval_data_train_ToyCar.zip"),
        ]:
            zip_path = os.path.join(cache_dir, name)
            if not os.path.exists(zip_path):
                print(f"  Downloading {name}...")
                urllib.request.urlretrieve(url, zip_path)
                size_mb = os.path.getsize(zip_path) / 1024 / 1024
                print(f"  Downloaded: {size_mb:.1f} MB")
            print(f"  Extracting {name}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(os.path.join(cache_dir, "dev_data"))

    # Spectrogram parameters (matching upstream baseline.yaml)
    n_mels = 128
    frames = 5
    n_fft = 1024
    hop_length = 512
    power = 2.0

    if not os.path.exists(labels_path):
        print(f"  ERROR: y_labels.csv not found: {labels_path}")
        return False

    with open(labels_path) as f:
        entries = list(csv.reader(f))

    os.makedirs(eval_dir, exist_ok=True)
    converted = 0
    missing = 0

    print(f"  Converting {len(entries)} AD samples...")
    for entry in entries:
        bin_name = entry[0].strip()
        bin_path = os.path.join(eval_dir, bin_name)
        if os.path.exists(bin_path):
            converted += 1
            continue

        # Derive WAV path from bin name
        # e.g. "normal_id_01_00000003_hist_librosa.bin"
        #   -> "normal_id_01_00000003.wav" in test/ directory
        wav_stem = bin_name.replace("_hist_librosa.bin", ".wav")
        wav_path = os.path.join(toycar_dir, "test", wav_stem)
        if not os.path.exists(wav_path):
            missing += 1
            if missing <= 3:
                print(f"  WARNING: WAV not found: {wav_path}")
            continue

        # Compute mel spectrogram (matching upstream common.py)
        y, sr = librosa.load(wav_path, sr=None, mono=False)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, power=power
        )
        log_mel = 20.0 / power * np.log10(mel_spec + sys.float_info.epsilon)

        # Take central part (frames 50:250) matching upstream
        log_mel = log_mel[:, 50:250]

        # Save as float32 (transposed to match upstream bin format)
        np.swapaxes(log_mel, 0, 1).astype("float32").tofile(bin_path)
        converted += 1

    if missing > 3:
        print(f"  WARNING: {missing} WAV files not found")
    print(f"  Prepared {converted}/{len(entries)} .bin files")
    return converted == len(entries)


def prepare_dataset(benchmark, eval_dir):
    """Prepare evaluation dataset for the specified benchmark."""
    cfg = BENCHMARK_CONFIG[benchmark]
    if cfg["dataset_prep"] == "cifar10":
        return prepare_dataset_ic01(eval_dir)
    if cfg["dataset_prep"] == "vww":
        return prepare_dataset_vww01(eval_dir)
    if cfg["dataset_prep"] == "kws":
        return prepare_dataset_kws01(eval_dir)
    if cfg["dataset_prep"] == "ad":
        return prepare_dataset_ad01(eval_dir)
    return False


def run_benchmark(args):
    """Run the benchmark using upstream runner components."""
    benchmark = args.benchmark
    eval_dir = os.path.join(EVAL_BASE_DIR, benchmark)
    results_dir = os.path.join(
        SUBMISSION_BASE_DIR, BENCHMARK_CONFIG[benchmark]["name"]
    )

    print(f"[1/4] Preparing dataset for {benchmark}...")
    if not prepare_dataset(benchmark, eval_dir):
        return 1

    # Runner's script.py creates sessions/ in cwd on import.
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)

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

    bench_config = test_scripts.get(benchmark)
    if not bench_config:
        print(f"ERROR: {benchmark} not found in {test_script_path}")
        return 1

    print(f"[2/4] Test config: {bench_config}")

    # Connect to K230
    print(f"[3/4] Connecting to {args.port}...")
    port = K230SerialDevice(args.port, args.baud, echo=args.echo)

    with port:
        # Launch DUT
        print("  Launching DUT on K230...")
        port.write_line("\x03")
        time.sleep(0.5)
        while not port._message_queue.empty():
            port._message_queue.get_nowait()
        port.write("\r")
        time.sleep(0.5)
        while not port._message_queue.empty():
            port._message_queue.get_nowait()

        kmodel_path = f"/sharefs/mlperf_tiny/{benchmark}.kmodel"
        dut_cmd = f"/sharefs/mlperf_tiny/mlperf_tiny {kmodel_path}"
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
        dut._model = benchmark
        dut._profile = "ULPMark for tinyML Firmware V0.0.1"

        # Load dataset
        dataset = DataSet(eval_dir, bench_config["truth_file"])
        print(f"  Dataset: {dataset.get_num_files()} files")

        # Run script
        print(f"\n[4/4] Running {args.mode_name} benchmark ({benchmark})...")
        script = Script(bench_config)
        result = script.run(None, dut, dataset, args.mode)

        # Save results
        os.makedirs(results_dir, exist_ok=True)
        import json
        results_path = os.path.join(results_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")

        # Exit DUT
        port.write_line("exit")
        time.sleep(1)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run MLPerf Tiny benchmark on K230"
    )
    parser.add_argument(
        "--benchmark", required=True,
        choices=list(BENCHMARK_CONFIG.keys()),
        help="Benchmark to run",
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
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Working directory for runner output (sessions/, etc.)",
    )
    args = parser.parse_args()

    mode_names = {"a": "accuracy", "p": "performance"}
    args.mode_name = mode_names[args.mode]

    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
