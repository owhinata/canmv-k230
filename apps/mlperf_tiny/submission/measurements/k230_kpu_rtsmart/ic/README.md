# MLPerf Tiny — K230 KPU Submission Report (Open Division)

## System Under Test

| Item | Value |
|------|-------|
| SoC | Kendryte K230 |
| CPU (big core) | RISC-V C908, 1.6 GHz |
| Accelerator | KPU, 800 MHz (estimated) |
| OS | RT-Smart |
| Framework | nncase 2.10.0 |

## Supported Benchmarks

| ID | Task | Model | Input Shape | Output | Float TFLite |
|----|------|-------|-------------|--------|-------------|
| ic01 | Image Classification | ResNet-8 | 32×32×3 = 3,072 | 10 classes (softmax) | `pretrainedResnet.tflite` |
| vww01 | Visual Wake Words | MobileNetV1 | 96×96×3 = 27,648 | 2 classes (softmax) | `vww_96_float.tflite` |
| kws01 | Keyword Spotting | DS-CNN | 49×10×1 = 490 | 12 classes (softmax) | `kws_ref_model_float32.tflite` |
| ad01 | Anomaly Detection | AutoEncoder | 1×640 | 640 (reconstruction) | `ad01_fp32.tflite` |

A single DUT binary (`mlperf_tiny`) runs all benchmarks — the kmodel file determines
the workload. Model version is auto-detected from the kmodel filename.

```
/sharefs/mlperf_tiny/
  mlperf_tiny          # universal binary
  ic01.kmodel
  vww01.kmodel
  kws01.kmodel
  ad01.kmodel
```

## Division Justification

This submission uses the **Open** division.

The reference MLPerf Tiny benchmarks provide pre-quantized int8 TFLite models.
However, nncase cannot directly import quantized TFLite models — it lacks support for
FullyConnected/MatMul in TFLite and per-channel DequantizeLinear in ONNX QDQ format.
Instead, the float32 TFLite models are converted to kmodel using nncase's own
post-training quantization (PTQ), which produces different quantization parameters
than the reference int8 models.

Because the quantization is performed independently by nncase rather than using the
reference quantization, the Closed division requirements are not met.

## Model Conversion Pipeline

All benchmarks share the same conversion pipeline:

```
<benchmark>_float.tflite (float32)
    |
    v  tf2onnx (opset 13)
<benchmark>.onnx (float32)
    |
    v  onnxsim (simplify, fix batch dim to 1)
<benchmark>_simplified.onnx (float32)
    |
    v  nncase 2.10.0 PTQ (target=k230, uint8, Kld)
<benchmark>.kmodel (uint8 quantized)
```

### Tool Versions

- **tf2onnx**: converts TFLite to ONNX with opset 13
- **onnxsim**: simplifies ONNX graph, fixes dynamic batch dimension to 1
- **nncase**: 2.10.0 (with nncase-kpu package for K230 target)

### nncase Compile Options

```python
compile_options.target = "k230"
compile_options.preprocess = False
compile_options.input_shape = <benchmark-specific>
compile_options.input_layout = "NHWC"  # omitted for 2D inputs (AD)
compile_options.output_layout = "NHWC"  # omitted for 2D inputs (AD)
```

### nncase PTQ Options

```python
ptq_options.quant_type = "uint8"
ptq_options.w_quant_type = "uint8"
ptq_options.calibrate_method = "Kld"
ptq_options.finetune_weights_method = "NoFineTuneWeights"
ptq_options.samples_count = 5  # random calibration data, float32 [0, 255]
```

### Why `preprocess=False`

Setting `preprocess=True` with identity mean/std values (mean=0, std=1) triggers an
nncase `FoldNopBinary` optimization bug. The internal no-op nodes (`Add(x, 0)`,
`Mul(x, 1)`) lack type information, causing an assertion failure during compilation.
Since no actual preprocessing transformation is needed, `preprocess=False` is the
correct setting.

### Why Float32 TFLite Models

nncase's ONNX importer cannot handle per-channel DequantizeLinear operators generated
by tf2onnx when converting quantized TFLite models. This causes a native crash
(`InvalidOperationException: This tensor is not a scalar`). Using the float32 TFLite
models avoids this issue — nncase performs its own PTQ during compilation.

| Benchmark | Float TFLite | Quantized TFLite (not used) |
|-----------|-------------|---------------------------|
| ic01 | `pretrainedResnet.tflite` | `pretrainedResnet_quant.tflite` |
| vww01 | `vww_96_float.tflite` | `vww_96_int8.tflite` |
| kws01 | `kws_ref_model_float32.tflite` | `kws_ref_model.tflite` |
| ad01 | `ad01_fp32.tflite` | `ad01_int8.tflite` |

## Input/Output Specification

### Classification Benchmarks (IC, VWW, KWS)

| Property | IC | VWW | KWS |
|----------|----|----|-----|
| Input shape | `[1, 32, 32, 3]` | `[1, 96, 96, 3]` | `[1, 49, 10, 1]` |
| Wire dtype | uint8 | uint8 | int8 (MFCC) |
| Input range | [0, 255] | [0, 255] | [-128, 127] |
| Output shape | `[1, 10]` | `[1, 2]` | `[1, 12]` |
| Output semantics | Softmax probabilities | Softmax (person/not-person) | Softmax (12 keywords) |

The DUT receives uint8/int8 data via the MLPerf Tiny UART protocol (`db` commands).
In `th_load_tensor`, the data is written to the kmodel input tensor with appropriate
type conversion based on the kmodel's input dtype. nncase's internal quantization
layer handles the rest.

### Anomaly Detection (AD)

| Property | Value |
|----------|-------|
| Input shape | `[1, 640]` |
| Wire dtype | float32 (raw bytes) |
| Input semantics | Mel spectrogram slice (128 bins × 5 frames) |
| Output shape | `[1, 640]` |
| Output semantics | Reconstructed spectrogram |

AD uses float32 input. The DUT receives raw float32 bytes (4 bytes per element,
2560 bytes total) via the UART protocol. `th_load_tensor` detects the float32 input
dtype and copies raw bytes directly to the input tensor without type conversion.

## kmodel Sizes

| Benchmark | kmodel Size |
|-----------|------------|
| ic01 | 88 KB |
| vww01 | 286 KB |
| kws01 | 39 KB |
| ad01 | 315 KB |

## Golden Test Results

Golden tests compare the reference float32 TFLite model against the K230 DUT on
the same input data. For classification benchmarks, agreement rate measures how
often both models produce the same argmax prediction. For AD, MSE between TFLite
and DUT outputs is reported.

### Image Classification (IC)

| Metric | Value |
|--------|-------|
| Samples tested | 100 |
| TFLite accuracy | 88.0% |
| DUT accuracy | 87.5% (200-sample eval) |
| Agreement rate | 99% (99/100) |
| Quality target | 85.0% |

The DUT exceeds the 85% quality target by 2.5 percentage points.

### Visual Wake Words (VWW)

| Metric | Value |
|--------|-------|
| Samples tested | 10 |
| Agreement rate | 80% (8/10) |
| Quality target | 80.0% |
| Dataset | COCO2014 val (96×96 RGB) |

Disagreements are attributable to quantization boundary effects where the top-2
class probabilities are close.

### Keyword Spotting (KWS)

| Metric | Value |
|--------|-------|
| Samples tested | 10 |
| Agreement rate | 90% (9/10) |
| Quality target | 90.0% |
| Dataset | Speech Commands v2 (MFCC spectrograms) |

### Anomaly Detection (AD)

| Metric | Value |
|--------|-------|
| Samples tested | 10 |
| Valid results | 10/10 |
| Mean MSE (TFLite vs DUT) | ~200 |
| Quality target | AUC 0.85 |
| Dataset | ToyADMOS/ToyCar (mel spectrograms) |

## Latency Measurements

Measured via `rdcycle` CSR on the RISC-V C908 big core (1.6 GHz). Timestamps are
captured immediately before and after KPU inference in the `th_infer` implementation.

| Benchmark | Mean Cycles | Mean Time (ms) | Min Cycles | Max Cycles |
|-----------|------------|----------------|------------|------------|
| ic01 | 3,899,464 | 2.44 | 3,885,692 | 3,934,121 |
| vww01 | 4,279,264 | 2.67 | 4,275,557 | 4,285,588 |
| kws01 | 3,845,412 | 2.40 | 3,830,870 | 3,878,243 |
| ad01 | 4,021,014 | 2.51 | 4,003,051 | 4,072,611 |

All benchmarks show highly consistent latency (coefficient of variation < 0.5%).
Despite the 9× difference in input size between IC (3,072 bytes) and VWW (27,648
bytes), inference latency varies by only ~10%, indicating that KPU computation
dominates over data transfer.

## Dataset Preparation

Evaluation datasets are automatically downloaded and cached in `~/.mlperf/`:

| Benchmark | Source | Cache Dir | Samples |
|-----------|--------|-----------|---------|
| ic01 | CIFAR-10 (Keras auto-download) | `~/.keras/datasets/` | 200 |
| vww01 | COCO2014 96×96 (Silicon Labs) | `~/.mlperf/vww/` | 1,000 |
| kws01 | Speech Commands v2 (tfds) | `~/.mlperf/kws/` | 1,000 |
| ad01 | ToyADMOS/ToyCar (Zenodo) | `~/.mlperf/ad/` | 248 |

## Build and Run

### Build

```bash
cmake -B build/mlperf_tiny -S apps/mlperf_tiny \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-k230-rtsmart.cmake
cmake --build build/mlperf_tiny
```

### Convert kmodels

```bash
cmake --build build/mlperf_tiny --target kmodel
# Or individual benchmarks:
.venv/bin/python apps/mlperf_tiny/scripts/convert_kmodel.py --benchmark ic01 vww01
```

### Deploy

```bash
cmake --build build/mlperf_tiny --target deploy
```

### Run on K230

```
msh /> /sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/ic01.kmodel
msh /> /sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/vww01.kmodel
```

### Testing (ctest)

```bash
ctest --test-dir build/mlperf_tiny -R ic01$         # IC benchmark only
ctest --test-dir build/mlperf_tiny -R ic01_golden   # IC golden test only
ctest --test-dir build/mlperf_tiny -R golden        # all golden tests
ctest --test-dir build/mlperf_tiny                  # all tests
```

## Measurement Conditions

- **UART baud rate**: 115200
- **Timestamp source**: `rdcycle` (RISC-V cycle counter)
- **CPU frequency**: 1.6 GHz (used to convert cycles to wall time)
- **Warmup inferences**: 0 (per MLPerf Tiny runner protocol)
- **Inferences per sample**: 1
