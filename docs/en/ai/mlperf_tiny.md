# MLPerf Tiny — K230 DUT Implementation

A DUT (Device Under Test) implementation for measuring K230 KPU inference performance using the [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmark framework.

Currently supports **Image Classification (CIFAR-10, ResNet-8)**.

!!! note "K230 and MLPerf Tiny"
    MLPerf Tiny typically targets 10-250MHz / <50mW class MCUs. The K230 falls outside this category, but by implementing a DUT conforming to the submitter API, we can reuse the standard measurement procedures provided by the official harness.

## Prerequisites

- K230 SDK must be built (toolchain extracted, MPP libraries compiled)
- SDK placed at `k230_sdk/` in the repository root
- CMake 3.16 or later
- UART connection (115200 bps) — for communication with the MLPerf Tiny legacy harness

!!! note "Building the SDK"
    For K230 SDK build instructions, see [SDK Build](../development/sdk_build.md).

## Overall Workflow

```
[Host PC]                         [K230 DUT]
                                    |
1. git submodule update             |
2. cmake configure/build            |
3. deploy (SCP)                     |
                                    |
4. Start DUT (UART)          -->  main loop
                                    |
5. runner (Python)           -->  UART command processing
   name%                      <--  m-name-dut-[...]
   db load N%                 <--  m-[Expecting N bytes]
   db HEXDATA%                <--  m-load-done
   infer N W%                 <--  m-results-[...]
   results%                   <--  m-results-[...]
```

## Build Instructions

### 1. Fetch submodule

```bash
git submodule update --init mlperf_tiny
```

### 2. Configure

```bash
cmake -B build/mlperf_tiny -S apps/mlperf_tiny \
  -DCMAKE_TOOLCHAIN_FILE="$(pwd)/cmake/toolchain-k230-rtsmart.cmake"
```

### 3. Build

```bash
cmake --build build/mlperf_tiny
```

### 4. Verify

```bash
file build/mlperf_tiny/mlperf_tiny
```

Expected output:

```
mlperf_tiny: ELF 64-bit LSB executable, UCB RISC-V, ...
```

## Deploying and Running on K230

### deploy target

Specify the kmodel file path and deploy:

```bash
cmake build/mlperf_tiny -DMLPERF_KMODEL=/path/to/model.kmodel
cmake --build build/mlperf_tiny --target deploy
```

### Manual transfer

```bash
scp build/mlperf_tiny/mlperf_tiny root@<K230_IP>:/sharefs/mlperf_tiny/
scp /path/to/model.kmodel root@<K230_IP>:/sharefs/mlperf_tiny/model.kmodel
```

### Running on K230 bigcore (msh)

```
msh /> /sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/model.kmodel
```

Expected output on successful startup:

```
m-timestamp-mode-performance
m-lap-us-XXXXXXXX
m-init-done
m-ready
```

## Manual UART Command Testing

Connect to the bigcore serial port (`/dev/ttyACM1`, 115200 bps) using minicom or similar, and send the following commands (each terminated with `%`):

| Command | Description | Expected Response |
|---------|-------------|-------------------|
| `name%` | Show device name | `m-name-dut-[unspecified]` |
| `profile%` | Show profile | `m-profile-[...]` / `m-model-[ic01]` |
| `help%` | Show help | Command list |
| `db load 3072%` | Allocate input buffer (32x32x3) | `m-[Expecting 3072 bytes]` |
| `infer 1 0%` | Run 1 inference (0 warmup) | `m-results-[...]` |
| `results%` | Show last results | `m-results-[...]` |

## Runner-based Measurement

Use the MLPerf Tiny Python runner for automated measurement:

```bash
cd mlperf_tiny/benchmark/runner
python main.py --port /dev/ttyACM1 --baud 115200
```

!!! warning "Runner requirements"
    The runner is part of the legacy UART harness. MLCommons is transitioning to a new runner, so procedures may change in future versions.

## CMake Targets

| Target | Command | Description |
|--------|---------|-------------|
| (default) | `cmake --build build/mlperf_tiny` | Build C++ binary |
| `deploy` | `cmake --build build/mlperf_tiny --target deploy` | Build + SCP transfer to K230 |
| `run` | `cmake --build build/mlperf_tiny --target run` | Execute on K230 via serial |

## CMake Options

| Variable | Default | Description |
|----------|---------|-------------|
| `MLPERF_BENCHMARK` | `ic` | Benchmark type |
| `MLPERF_KMODEL` | (empty) | Path to kmodel file for deployment |

## Source Files

| File | Description |
|------|-------------|
| `src/main.cc` | Entry point — kmodel path argument, UART main loop |
| `src/submitter_implemented.cc` | K230/nncase implementation of th_* functions |

## Troubleshooting

### UART Communication Failure

- Verify baud rate is 115200 bps
- Verify using the bigcore serial port (`/dev/ttyACM1`)
- Ensure minicom/picocom is not occupying the port

### kmodel Load Failure

- Verify the kmodel file path is correct
- Check nncase version compatibility with the kmodel

### VB Initialization Failure

- The current implementation omits VB initialization
- If the nncase runtime requires VB, add VB configuration to `InitPlatform()` in `submitter_implemented.cc`

## kmodel Conversion

Use `convert_kmodel.py` to convert a TFLite model to a kmodel for the K230 KPU. The conversion pipeline is a two-stage process: TFLite → ONNX → kmodel.

### Install dependencies

```bash
pip install tf2onnx tensorflow-cpu onnxsim nncase nncase-kpu
```

### Run conversion

```bash
python convert_kmodel.py
```

This script performs the following steps:

1. Converts the TFLite model to ONNX format using `tf2onnx`
2. Optimizes the ONNX model with `onnxsim`
3. Compiles the ONNX model to kmodel using the `nncase` compiler (targeting K230 KPU)

The generated kmodel file can be deployed to the K230 using the deploy procedure described above.

## Golden Inference Test

`golden_test.py` compares TFLite reference inference results against K230 DUT inference results to verify the correctness of model conversion and device implementation.

### Usage

```bash
python golden_test.py
```

### How it works

1. Retrieves input images from the CIFAR-10 test dataset
2. Runs reference inference using the TFLite interpreter
3. Automatically launches the K230 DUT and sends the same inputs via UART
4. Compares DUT inference results against reference results
5. Reports accuracy and agreement metrics

The DUT launch and communication are fully automated — no manual DUT startup is required.

## Runner-based Benchmark

`run_benchmark.py` runs a standard accuracy benchmark using the upstream MLPerf Tiny runner.

### Usage

```bash
python run_benchmark.py
```

### How it works

1. Generates an IC (Image Classification) evaluation dataset from CIFAR-10
2. Runs a 200-sample accuracy benchmark
3. Performs measurement conforming to the upstream MLPerf Tiny runner protocol

## Results Summary

| Metric | Result | Target |
|--------|--------|--------|
| Accuracy | 87.5% | 85% |
| Latency | ~2.3ms | — |
| Agreement with reference | 99% | — |

- Accuracy achieved **87.5%**, exceeding the 85% target
- Inference latency of approximately **2.3ms** demonstrates the K230 KPU's high-speed inference capability
- **99%** agreement with the TFLite reference confirms the correctness of the kmodel conversion and DUT implementation
