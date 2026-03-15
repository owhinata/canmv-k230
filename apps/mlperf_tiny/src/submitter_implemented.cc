// Copyright 2024 The MLPerf Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
//
// K230/nncase implementation of th_* functions for MLPerf Tiny.
// Supports all benchmarks: IC, VWW, KWS, AD via runtime tensor info.

#include "api/submitter_implemented.h"

#include <signal.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>

#include "api/internally_implemented.h"

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

const char* kmodel_path_ = nullptr;

namespace {

using namespace nncase::runtime;  // NOLINT(build/namespaces)

interpreter interp;

// Element counts — determined at runtime from kmodel tensor descriptors
static int input_elements_ = 0;
static int output_elements_ = 0;

// Raw output buffer — sized to MAX_DB_INPUT_SIZE to cover all benchmarks
// (largest input is VWW: 96*96*3 = 27648 = MAX_DB_INPUT_SIZE)
float output_float[MAX_DB_INPUT_SIZE];

// ---------------------------------------------------------------------------
// Internal helpers — platform / model / tensor lifecycle
// ---------------------------------------------------------------------------

void InitPlatform() {
  // Intentionally empty — VB not required for initial bring-up.
  // If nncase requires VB pools, add kd_mpi_vb_set_config() here.
}

void LoadKmodel() {
  if (kmodel_path_ == nullptr) {
    printf("ERROR: kmodel path not set\n");
    return;
  }

  std::ifstream ifs(kmodel_path_, std::ios::binary);
  if (!ifs) {
    printf("ERROR: cannot open kmodel: %s\n", kmodel_path_);
    return;
  }

  interp.load_model(ifs).expect("load_model failed");
}

void PrepareTensors() {
  // Compute input element count from shape
  input_elements_ = 1;
  auto in_shape = interp.input_shape(0);
  for (size_t d = 0; d < in_shape.size(); d++) {
    input_elements_ *= static_cast<int>(in_shape[d]);
  }

  // Compute output element count from shape
  output_elements_ = 1;
  auto out_shape = interp.output_shape(0);
  for (size_t d = 0; d < out_shape.size(); d++) {
    output_elements_ *= static_cast<int>(out_shape[d]);
  }

  printf("Model: input_elements=%d, output_elements=%d\n",
         input_elements_, output_elements_);

  for (size_t i = 0; i < interp.inputs_size(); i++) {
    auto desc = interp.input_desc(i);
    auto shape = interp.input_shape(i);
    auto tensor =
        host_runtime_tensor::create(desc.datatype, shape,
                                    nncase::runtime::hrt::pool_shared)
            .expect("cannot create input tensor");
    interp.input_tensor(i, tensor).expect("cannot set input tensor");
  }
}

// ---------------------------------------------------------------------------
// Output helpers — separate raw read from formatted output
// ---------------------------------------------------------------------------

void ReadOutputTensorRaw() {
  auto out_tensor = interp.output_tensor(0).expect("cannot get output tensor");
  auto out_map =
      std::move(nncase::runtime::hrt::map(out_tensor,
                                           nncase::runtime::map_read)
                    .unwrap_or_throw());
  auto out_span = out_map.buffer();

  auto desc = interp.output_desc(0);

  if (desc.datatype == nncase::typecode_t::dt_int8) {
    const int8_t* data = reinterpret_cast<const int8_t*>(out_span.data());
    for (int i = 0; i < output_elements_; i++) {
      output_float[i] = static_cast<float>(data[i]);
    }
  } else if (desc.datatype == nncase::typecode_t::dt_float32) {
    const float* data = reinterpret_cast<const float*>(out_span.data());
    for (int i = 0; i < output_elements_; i++) {
      output_float[i] = data[i];
    }
  } else {
    printf("WARNING: unsupported output dtype\n");
    for (int i = 0; i < output_elements_; i++) {
      output_float[i] = 0.0f;
    }
  }
}

void FormatResultsForRunner() {
  th_printf("m-results-[");
  for (int i = 0; i < output_elements_; i++) {
    th_printf("%0.3f", static_cast<double>(output_float[i]));
    if (i < output_elements_ - 1) {
      th_printf(",");
    }
  }
  th_printf("]\r\n");
}

// ---------------------------------------------------------------------------
// Transport layer
// ---------------------------------------------------------------------------

void TransportWriteFmt(const char* fmt, va_list ap) {
  vprintf(fmt, ap);
  fflush(stdout);
}

int TransportReadChar() { return getchar(); }

}  // namespace

// ===========================================================================
// th_* API implementation
// ===========================================================================

void th_load_tensor() {
  // Get input tensor info
  auto in_tensor = interp.input_tensor(0).expect("cannot get input tensor");
  auto in_map =
      std::move(nncase::runtime::hrt::map(in_tensor,
                                           nncase::runtime::map_write)
                    .unwrap_or_throw());
  auto in_span = in_map.buffer();
  auto desc = interp.input_desc(0);

  if (desc.datatype == nncase::typecode_t::dt_float32) {
    // Float32 input (e.g. AD): read raw float32 bytes from buffer
    uint8_t raw_buf[MAX_DB_INPUT_SIZE];
    size_t expected = input_elements_ * sizeof(float);
    size_t bytes = ee_get_buffer(raw_buf, expected);
    if (bytes != expected) {
      th_printf("Input db has %d bytes, expected %d\n",
                static_cast<int>(bytes), static_cast<int>(expected));
      return;
    }
    memcpy(in_span.data(), raw_buf, expected);
  } else {
    // Quantized input (IC/VWW/KWS): read uint8 elements
    uint8_t input_quantized[MAX_DB_INPUT_SIZE];
    size_t bytes =
        ee_get_buffer(input_quantized, input_elements_ * sizeof(uint8_t));
    if (static_cast<int>(bytes / sizeof(uint8_t)) != input_elements_) {
      th_printf("Input db has %d elements, expected %d\n",
                static_cast<int>(bytes / sizeof(uint8_t)), input_elements_);
      return;
    }

    if (desc.datatype == nncase::typecode_t::dt_int8) {
      // Convert uint8 [0,255] -> int8 [-128,127]
      int8_t* dst = reinterpret_cast<int8_t*>(in_span.data());
      for (int i = 0; i < input_elements_; i++) {
        if (input_quantized[i] <= 127) {
          dst[i] = static_cast<int8_t>(input_quantized[i]) - 128;
        } else {
          dst[i] = static_cast<int8_t>(input_quantized[i] - 128);
        }
      }
    } else if (desc.datatype == nncase::typecode_t::dt_uint8) {
      memcpy(in_span.data(), input_quantized, input_elements_);
    } else {
      th_printf("WARNING: unsupported input dtype, copying raw bytes\n");
      size_t copy_len = static_cast<size_t>(input_elements_) < in_span.size()
                            ? static_cast<size_t>(input_elements_)
                            : in_span.size();
      memcpy(in_span.data(), input_quantized, copy_len);
    }
  }
}

void th_results() {
  ReadOutputTensorRaw();
  FormatResultsForRunner();
}

void th_infer() { interp.run().expect("inference failed"); }

void th_timestamp(void) {
  unsigned long cycles = 0;  // NOLINT(runtime/int)
#if defined(__riscv)
  asm volatile("rdcycle %0" : "=r"(cycles));
#endif
  th_printf(EE_MSG_TIMESTAMP, cycles);
}

void th_printf(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  TransportWriteFmt(fmt, args);
  va_end(args);
}

char th_getchar() { return static_cast<char>(TransportReadChar()); }

// --- optional API ---

void th_serialport_initialize(void) {
  setvbuf(stdin, nullptr, _IONBF, 0);
  setvbuf(stdout, nullptr, _IONBF, 0);
}

void th_timestamp_initialize(void) {
  th_printf(EE_MSG_TIMESTAMP_MODE);
  th_timestamp();
}

void th_final_initialize(void) {
  InitPlatform();
  LoadKmodel();
  PrepareTensors();
}

void th_pre() {}
void th_post() {}

extern volatile sig_atomic_t dut_running_;

void th_command_ready(char volatile* p_command) {
  if (strncmp(const_cast<char*>(p_command), "exit", 4) == 0) {
    dut_running_ = 0;
    return;
  }
  ee_serial_command_parser_callback(const_cast<char*>(p_command));
}

// --- libc wrappers ---

int th_strncmp(const char* str1, const char* str2, size_t n) {
  return strncmp(str1, str2, n);
}

char* th_strncpy(char* dest, const char* src, size_t n) {
  return strncpy(dest, src, n);
}

size_t th_strnlen(const char* str, size_t maxlen) {
  return strnlen(str, maxlen);
}

char* th_strcat(char* dest, const char* src) {
  return strncat(dest, src, strlen(src));  // NOLINT(runtime/printf)
}

char* th_strtok(char* str1, const char* sep) { return strtok(str1, sep); }

int th_atoi(const char* str) { return atoi(str); }

void* th_memset(void* b, int c, size_t len) { return memset(b, c, len); }

void* th_memcpy(void* dst, const void* src, size_t n) {
  return memcpy(dst, src, n);
}

int th_vprintf(const char* format, va_list ap) { return vprintf(format, ap); }
