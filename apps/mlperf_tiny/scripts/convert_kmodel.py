"""Convert MLPerf Tiny TFLite model to K230 kmodel.

Pipeline: TFLite (float32) -> ONNX (tf2onnx) -> simplify (onnxsim) -> kmodel (nncase)

The float TFLite model is used because:
- nncase TFLite importer does not support FullyConnected/MatMul
- nncase ONNX importer does not support per-channel DequantizeLinear (QDQ format)
nncase performs its own PTQ quantization during compilation.

Note: compile_options.preprocess must be False. Setting preprocess=True with
identity mean/std (0/1) triggers an nncase FoldNopBinary bug where internal
nop nodes (Add(x,0), Mul(x,1)) lack type info, causing an assertion failure.

Usage:
  python apps/mlperf_tiny/scripts/convert_kmodel.py
  python apps/mlperf_tiny/scripts/convert_kmodel.py -o /path/to/output.kmodel

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
TFLITE_PATH = os.path.join(
    REPO_ROOT,
    "mlperf_tiny/benchmark/training/image_classification/"
    "trained_models/pretrainedResnet.tflite",
)

INPUT_H, INPUT_W, INPUT_C = 32, 32, 3


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


def onnx_to_kmodel(onnx_path, kmodel_path):
    """Convert ONNX to kmodel using nncase PTQ."""
    compile_options = nncase.CompileOptions()
    compile_options.target = "k230"
    compile_options.dump_ir = False
    compile_options.dump_asm = False
    # preprocess=False to avoid FoldNopBinary bug with identity mean/std
    compile_options.preprocess = False
    compile_options.input_shape = [1, INPUT_H, INPUT_W, INPUT_C]
    compile_options.input_layout = "NHWC"
    compile_options.output_layout = "NHWC"

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

    # Random calibration data (float32, matching model input range)
    samples = [
        np.random.rand(1, INPUT_H, INPUT_W, INPUT_C).astype(np.float32)
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert MLPerf Tiny TFLite model to K230 kmodel"
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(SCRIPT_DIR, "..", "model.kmodel"),
        help="Output kmodel path",
    )
    parser.add_argument(
        "--tflite",
        default=TFLITE_PATH,
        help="Input TFLite model path (float32, not quantized)",
    )
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)

    if not os.path.exists(args.tflite):
        print(f"ERROR: TFLite model not found: {args.tflite}")
        print("Run: git submodule update --init mlperf_tiny")
        return 1

    print(f"Input:  {args.tflite}")
    print(f"Output: {output_path}")

    # Step 1: TFLite -> ONNX (simplified)
    print("\n[1/2] TFLite -> ONNX")
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx_path = tmp.name

    try:
        if not tflite_to_onnx(args.tflite, onnx_path):
            return 1

        # Step 2: ONNX -> kmodel
        print("\n[2/2] ONNX -> kmodel")
        if not onnx_to_kmodel(onnx_path, output_path):
            return 1
    finally:
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)

    print(f"\nDone: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
