"""Convert MLPerf Tiny TFLite models to K230 kmodel.

Pipeline: TFLite (float32) -> ONNX (tf2onnx) -> simplify (onnxsim) -> kmodel (nncase)

The float TFLite model is used because:
- nncase TFLite importer does not support FullyConnected/MatMul
- nncase ONNX importer does not support per-channel DequantizeLinear (QDQ format)
nncase performs its own PTQ quantization during compilation.

Note: compile_options.preprocess must be False. Setting preprocess=True with
identity mean/std (0/1) triggers an nncase FoldNopBinary bug where internal
nop nodes (Add(x,0), Mul(x,1)) lack type info, causing an assertion failure.

Usage:
  python convert_kmodel.py --benchmark ic01
  python convert_kmodel.py --benchmark ic01 vww01 kws01 ad01
  python convert_kmodel.py --benchmark ic01 vww01 kws01 ad01 -o build/kmodels

Prerequisites:
  pip install tf2onnx tensorflow-cpu onnxsim nncase nncase-kpu
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnxsim import simplify

import nncase

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TRAINING_DIR = os.path.join(REPO_ROOT, "mlperf_tiny", "benchmark", "training")

BENCHMARKS = {
    "ic01": {
        "tflite": os.path.join(
            TRAINING_DIR,
            "image_classification/trained_models/pretrainedResnet.tflite",
        ),
        "shape": [1, 32, 32, 3],
    },
    "vww01": {
        "tflite": os.path.join(
            TRAINING_DIR,
            "visual_wake_words/trained_models/vww_96_float.tflite",
        ),
        "shape": [1, 96, 96, 3],
    },
    "kws01": {
        "tflite": os.path.join(
            TRAINING_DIR,
            "keyword_spotting/trained_models/kws_ref_model_float32.tflite",
        ),
        "shape": [1, 49, 10, 1],
    },
    "ad01": {
        "tflite": os.path.join(
            TRAINING_DIR,
            "anomaly_detection/trained_models/ad01_fp32.tflite",
        ),
        "shape": [1, 640],
    },
}


def tflite_to_onnx(tflite_path, onnx_path):
    """Convert TFLite to ONNX, fix batch dim, and simplify."""
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", tflite_path,
        "--output", onnx_path,
        "--opset", "13",
    ]
    print(f"  tf2onnx: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: tf2onnx failed:\n{result.stderr}")
        return False

    # Fix dynamic batch dim to 1 and simplify
    model = onnx.load(onnx_path)
    for inp in model.graph.input:
        inp.type.tensor_type.shape.dim[0].dim_value = 1
    for out in model.graph.output:
        out.type.tensor_type.shape.dim[0].dim_value = 1

    model_sim, check = simplify(model)
    if not check:
        print("WARNING: onnxsim simplification failed, using unsimplified model")
        model_sim = model

    onnx.save(model_sim, onnx_path)
    ops = sorted(set(n.op_type for n in model_sim.graph.node))
    print(f"  ONNX ops: {ops}")
    return True


def onnx_to_kmodel(onnx_path, kmodel_path, input_shape):
    """Convert ONNX to kmodel using nncase PTQ."""
    compile_options = nncase.CompileOptions()
    compile_options.target = "k230"
    compile_options.dump_ir = False
    compile_options.dump_asm = False
    # preprocess=False to avoid FoldNopBinary bug with identity mean/std
    compile_options.preprocess = False
    compile_options.input_shape = input_shape
    compile_options.input_layout = "NHWC" if len(input_shape) == 4 else ""
    compile_options.output_layout = "NHWC" if len(input_shape) == 4 else ""

    ptq_options = nncase.PTQTensorOptions()
    ptq_options.quant_type = "uint8"
    ptq_options.w_quant_type = "uint8"
    ptq_options.calibrate_method = "Kld"
    ptq_options.finetune_weights_method = "NoFineTuneWeights"
    ptq_options.dump_quant_error = False
    ptq_options.quant_scheme = ""
    ptq_options.quant_scheme_strict_mode = False
    ptq_options.export_quant_scheme = False
    ptq_options.export_weight_range_by_channel = False

    # Random calibration data (float32 [0,255], matching model input range)
    samples = [
        np.random.randint(0, 256, input_shape).astype(np.float32)
        for _ in range(5)
    ]
    ptq_options.samples_count = len(samples)
    ptq_options.set_tensor_data([samples])

    print("  Compiling ONNX -> kmodel...")
    compiler = nncase.Compiler(compile_options)
    with open(onnx_path, "rb") as f:
        compiler.import_onnx(f.read(), nncase.ImportOptions())
    compiler.use_ptq(ptq_options)
    compiler.compile()
    kmodel = compiler.gencode_tobytes()

    os.makedirs(os.path.dirname(os.path.abspath(kmodel_path)), exist_ok=True)
    with open(kmodel_path, "wb") as f:
        f.write(kmodel)

    print(f"  kmodel: {len(kmodel):,} bytes ({len(kmodel) / 1024:.1f} KB)")
    return True


def convert_benchmark(bench_id, output_dir):
    """Convert a single benchmark's TFLite model to kmodel."""
    cfg = BENCHMARKS[bench_id]
    tflite_path = cfg["tflite"]
    input_shape = cfg["shape"]
    kmodel_path = os.path.join(output_dir, f"{bench_id}.kmodel")

    print(f"\n{'='*60}")
    print(f"Benchmark: {bench_id}")
    print(f"  TFLite:  {tflite_path}")
    print(f"  Shape:   {input_shape}")
    print(f"  Output:  {kmodel_path}")

    if not os.path.exists(tflite_path):
        print(f"ERROR: TFLite model not found: {tflite_path}")
        print("Run: git submodule update --init mlperf_tiny")
        return False

    # Step 1: TFLite -> ONNX (simplified)
    print("\n  [1/2] TFLite -> ONNX")
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx_path = tmp.name

    try:
        if not tflite_to_onnx(tflite_path, onnx_path):
            return False

        # Step 2: ONNX -> kmodel
        print("\n  [2/2] ONNX -> kmodel")
        if not onnx_to_kmodel(onnx_path, kmodel_path, input_shape):
            return False
    finally:
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert MLPerf Tiny TFLite models to K230 kmodel"
    )
    parser.add_argument(
        "--benchmark", nargs="+",
        default=list(BENCHMARKS.keys()),
        choices=list(BENCHMARKS.keys()),
        help="Benchmarks to convert (default: all)",
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(SCRIPT_DIR, "..", "kmodels"),
        help="Output directory for kmodel files",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    success = []
    failed = []
    for bench_id in args.benchmark:
        if convert_benchmark(bench_id, output_dir):
            success.append(bench_id)
        else:
            failed.append(bench_id)

    print(f"\n{'='*60}")
    print(f"Done: {len(success)} succeeded, {len(failed)} failed")
    if success:
        print(f"  OK: {', '.join(success)}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
