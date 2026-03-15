# MLPerf Tiny — K230 DUT Implementation

A DUT (Device Under Test) implementation for measuring K230 KPU inference performance using the [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmark framework.

Supports **IC (Image Classification), VWW (Visual Wake Words), KWS (Keyword Spotting), and AD (Anomaly Detection)** — all four MLPerf Tiny benchmarks. A single binary + per-benchmark kmodel files runs all workloads.

!!! note "K230 and MLPerf Tiny"
    MLPerf Tiny typically targets 10-250MHz / <50mW class MCUs. The K230 falls outside this category, but by implementing a DUT conforming to the submitter API, we can reuse the standard measurement procedures provided by the official harness.

## Supported Benchmarks

| ID | Task | Model | Input Size | Output | Quality Target |
|----|------|-------|-----------|--------|---------------|
| ic01 | Image Classification | ResNet-8 | 32×32×3 = 3,072 | 10 classes | 85% |
| vww01 | Visual Wake Words | MobileNetV1 | 96×96×3 = 27,648 | 2 classes | 80% |
| kws01 | Keyword Spotting | DS-CNN | 49×10×1 = 490 | 12 classes | 90% |
| ad01 | Anomaly Detection | AutoEncoder | 1×640 | 640 (reconstruction) | AUC 0.85 |

## Prerequisites

- K230 SDK built (toolchain extracted, MPP libraries compiled)
- SDK placed at `k230_sdk/` in the repository root
- CMake 3.16 or later
- UART connection (115200 bps) — for MLPerf Tiny legacy harness communication

!!! note "Building the SDK"
    See [SDK Build](../development/sdk_build.md) for K230 SDK build instructions.

## Build

### 1. Get submodules

```bash
git submodule update --init mlperf_tiny
```

### 2. Configure and build

```bash
cmake -B build/mlperf_tiny -S apps/mlperf_tiny \
  -DCMAKE_TOOLCHAIN_FILE="$(pwd)/cmake/toolchain-k230-rtsmart.cmake"
cmake --build build/mlperf_tiny
```

### 3. Convert kmodels

Convert all benchmark kmodels:

```bash
cmake --build build/mlperf_tiny --target kmodel
```

Or individual benchmarks:

```bash
.venv/bin/python apps/mlperf_tiny/scripts/convert_kmodel.py --benchmark ic01
.venv/bin/python apps/mlperf_tiny/scripts/convert_kmodel.py --benchmark ic01 vww01 kws01 ad01
```

### 4. Deploy

Transfer binary + all kmodels to K230:

```bash
cmake --build build/mlperf_tiny --target deploy
```

## Running on K230

```
/sharefs/mlperf_tiny/
  mlperf_tiny          # universal binary
  ic01.kmodel
  vww01.kmodel
  kws01.kmodel
  ad01.kmodel
```

```
msh /> /sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/ic01.kmodel
msh /> /sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/vww01.kmodel
msh /> /sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/ad01.kmodel
```

Model version is auto-detected from the kmodel filename.

## Testing (ctest)

```bash
ctest --test-dir build/mlperf_tiny -R ic01$         # IC benchmark only
ctest --test-dir build/mlperf_tiny -R ic01_golden   # IC golden test only
ctest --test-dir build/mlperf_tiny -R golden        # all golden tests
ctest --test-dir build/mlperf_tiny                  # all tests
```

## Golden Tests

`golden_test.py` compares TFLite reference inference against K230 DUT inference on the same input data.

```bash
.venv/bin/python apps/mlperf_tiny/scripts/golden_test.py --benchmark ic01 -n 100
.venv/bin/python apps/mlperf_tiny/scripts/golden_test.py --benchmark vww01 -n 10
.venv/bin/python apps/mlperf_tiny/scripts/golden_test.py --benchmark kws01 -n 10
.venv/bin/python apps/mlperf_tiny/scripts/golden_test.py --benchmark ad01 -n 10
```

## Runner Benchmarks

`run_benchmark.py` runs standard benchmarks using the upstream MLPerf Tiny runner.

```bash
.venv/bin/python apps/mlperf_tiny/scripts/run_benchmark.py --benchmark ic01
.venv/bin/python apps/mlperf_tiny/scripts/run_benchmark.py --benchmark vww01 --mode p
```

## Datasets

Evaluation datasets are automatically downloaded and cached:

| Benchmark | Source | Cache | Samples |
|-----------|--------|-------|---------|
| ic01 | CIFAR-10 (Keras) | `~/.keras/datasets/` | 200 |
| vww01 | COCO2014 96×96 (Silicon Labs) | `~/.mlperf/vww/` | 1,000 |
| kws01 | Speech Commands v2 (tfds) | `~/.mlperf/kws/` | 1,000 |
| ad01 | ToyADMOS/ToyCar (Zenodo) | `~/.mlperf/ad/` | 248 |

## Results Summary

| Benchmark | Accuracy | Target | Latency | kmodel |
|-----------|----------|--------|---------|--------|
| IC (ic01) | 87.5% | 85% | ~2.4 ms | 88 KB |
| VWW (vww01) | — | 80% | ~2.7 ms | 286 KB |
| KWS (kws01) | — | 90% | ~2.4 ms | 39 KB |
| AD (ad01) | — | AUC 0.85 | ~2.5 ms | 315 KB |

- IC exceeds the 85% quality target with **87.5%** accuracy
- All benchmarks achieve **2.4–2.7 ms** latency, stable regardless of input size
- VWW/KWS/AD accuracy requires full evaluation dataset runs

## Source Files

| File | Description |
|------|-------------|
| `src/main.cc` | Entry point — kmodel path, model_version args, UART main loop |
| `src/submitter_implemented.cc` | th_* function implementation for K230/nncase (all benchmarks) |
| `src/internally_implemented.cpp` | Copied from submodule, TH_MODEL_VERSION made runtime |
| `scripts/convert_kmodel.py` | TFLite → kmodel conversion (all benchmarks) |
| `scripts/run_benchmark.py` | Runner-based benchmark execution |
| `scripts/golden_test.py` | TFLite vs DUT golden comparison test |
