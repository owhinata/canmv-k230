# Image Classification -- K230 KPU (Open Division)

## System Under Test

| Item | Value |
|------|-------|
| SoC | Kendryte K230 |
| CPU (big core) | RISC-V C908, 1.6 GHz |
| Accelerator | KPU, 800 MHz (estimated) |
| OS | RT-Smart |
| Framework | nncase 2.10.0 |

## Division Justification

This submission uses the **Open** division.

The reference MLPerf Tiny IC benchmark provides a pre-quantized int8 TFLite model.
However, nncase cannot directly import quantized TFLite models (it lacks support for
FullyConnected/MatMul in TFLite and per-channel DequantizeLinear in ONNX QDQ format).
Instead, the float32 TFLite model (`pretrainedResnet.tflite`) is converted to kmodel
using nncase's own post-training quantization (PTQ), which may produce different
quantization parameters than the reference int8 model.

Because the quantization is performed independently by nncase rather than using the
reference quantization, the Closed division requirements are not met.

## Model Conversion Pipeline

```
pretrainedResnet.tflite (float32, NHWC)
    |
    v  tf2onnx (opset 13)
model.onnx (float32, NHWC)
    |
    v  onnxsim (simplify, fix batch dim to 1)
model_simplified.onnx (float32, NHWC)
    |
    v  nncase 2.10.0 PTQ (target=k230)
model.kmodel (uint8 quantized)
```

### Tool Versions

- **tf2onnx**: converts TFLite to ONNX with opset 13
- **onnxsim**: simplifies ONNX graph, fixes dynamic batch dimension to 1
- **nncase**: 2.10.0 (with nncase-kpu package for K230 target)

### nncase Compile Options

```python
compile_options.target = "k230"
compile_options.preprocess = False
compile_options.input_shape = [1, 32, 32, 3]
compile_options.input_layout = "NHWC"
compile_options.output_layout = "NHWC"
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
Since no actual preprocessing transformation is needed (the model already expects
`[0, 255]` float32 input), `preprocess=False` is the correct setting.

## Input/Output Specification

### Input

| Property | Value |
|----------|-------|
| Shape | `[1, 32, 32, 3]` (NHWC) |
| Wire dtype | uint8 |
| Runtime dtype | float32 (cast in `th_load_tensor`) |
| Range | [0, 255] |

The DUT receives uint8 pixel data via the MLPerf Tiny UART protocol. In
`th_load_tensor`, the uint8 values are cast to float32 (preserving the [0, 255]
range) before being written to the kmodel input tensor. nncase's internal
quantization layer handles the float32-to-uint8 conversion using the PTQ
parameters baked into the kmodel.

### Output

| Property | Value |
|----------|-------|
| Shape | `[1, 10]` (NHWC) |
| Dtype | float32 |
| Semantics | Softmax probabilities (10 CIFAR-10 classes) |

## Quantization Equivalence Analysis

### Accuracy Comparison

| Model | Accuracy | Dataset |
|-------|----------|---------|
| Reference TFLite (float32) | 88.0% | CIFAR-10 test (first 100 samples) |
| DUT kmodel (nncase uint8 PTQ) | 87.5% | MLPerf Tiny IC eval (200 samples) |
| MLPerf Tiny quality target | 85.0% | -- |

The DUT achieves 87.5% accuracy on the 200-sample IC evaluation set, exceeding the
85% quality target by 2.5 percentage points.

### Agreement with Reference

A golden comparison test (`golden_test.py`) was run on the first 100 CIFAR-10 test
samples, comparing the argmax predictions of the reference float32 TFLite model
against the K230 DUT:

| Metric | Value |
|--------|-------|
| Samples tested | 100 |
| Agreement rate | 99% (99/100) |
| Disagreements | 1 |

The single disagreement is attributable to quantization boundary effects where the
top-2 class probabilities are close, and the uint8 quantization shifts the ranking.

### Methodology

- The reference TFLite model receives `float32 = uint8_pixel` (range [0, 255])
- The DUT receives the same uint8 bytes via serial; `th_load_tensor` casts to float32
- Both paths process identical input data
- Comparison is based on argmax of output predictions

## Latency Measurements

Measured via `rdcycle` CSR on the RISC-V C908 big core (1.6 GHz). Timestamps are
captured immediately before and after KPU inference in the `th_infer` implementation.

| Metric | Cycles | Time (ms) |
|--------|--------|-----------|
| Min | 3,740,011 | 2.338 |
| Max | 3,793,841 | 2.371 |
| Mean | 3,746,281 | 2.341 |
| Median | 3,745,782 | 2.341 |
| Std dev | 4,863 | 0.003 |

Based on 200 inference runs from the IC evaluation benchmark. Latency is highly
consistent (coefficient of variation < 0.2%).

## Measurement Conditions

- **UART baud rate**: 115200
- **Timestamp source**: `rdcycle` (RISC-V cycle counter)
- **CPU frequency**: 1.6 GHz (used to convert cycles to wall time)
- **Warmup inferences**: 0 (per MLPerf Tiny runner protocol)
- **Inferences per sample**: 1
