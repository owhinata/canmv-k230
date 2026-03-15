// Copyright 2024 The MLPerf Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
//
// K230 DUT entry point for MLPerf Tiny legacy UART harness.
// Send "exit%" via serial to quit cleanly.
//
// Usage: mlperf_tiny <kmodel_path> [model_version]
//   model_version: ic01, vww01, kws01, ad01
//   (default: auto-detect from filename)

#include <signal.h>

#include <cstdio>
#include <cstring>

#include "api/internally_implemented.h"
#include "api/submitter_implemented.h"

extern const char* kmodel_path_;

const char* model_version_ = "unknown";

volatile sig_atomic_t dut_running_ = 1;

static void HandleSignal(int sig) {
  if (sig == SIGINT) {
    dut_running_ = 0;
  }
}

// Auto-detect model version from kmodel filename (e.g. "ic01.kmodel" -> "ic01")
static const char* DetectModelVersion(const char* path) {
  static const char* known[] = {"ic01", "vww01", "kws01", "ad01"};
  const char* basename = strrchr(path, '/');
  basename = basename ? basename + 1 : path;
  for (const char* id : known) {
    if (strncmp(basename, id, strlen(id)) == 0) {
      return id;
    }
  }
  return "unknown";
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s <kmodel_path> [model_version]\n", argv[0]);
    printf("  model_version: ic01, vww01, kws01, ad01\n");
    return 1;
  }
  kmodel_path_ = argv[1];

  if (argc >= 3) {
    model_version_ = argv[2];
  } else {
    model_version_ = DetectModelVersion(argv[1]);
  }

  struct sigaction sa = {};
  sa.sa_handler = HandleSignal;
  sigfillset(&sa.sa_mask);
  sigaction(SIGINT, &sa, nullptr);

  ee_benchmark_initialize();

  while (dut_running_) {
    int c = th_getchar();
    // Filter CR/LF — RT-Smart msh delivers input line-buffered with
    // trailing CR.  The MLPerf protocol uses '%' as the sole terminator.
    if (c == '\r' || c == '\n') {
      continue;
    }
    ee_serial_callback(c);
  }

  printf("DUT exiting.\n");
  return 0;
}
