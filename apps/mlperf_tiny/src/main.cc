// Copyright 2024 The MLPerf Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
//
// K230 DUT entry point for MLPerf Tiny legacy UART harness.

#include <signal.h>

#include <cstdio>

#include "api/internally_implemented.h"
#include "api/submitter_implemented.h"

extern const char* g_kmodel_path;

static volatile sig_atomic_t running_ = 1;

static void HandleSignal(int sig) {
  if (sig == SIGINT) {
    running_ = 0;
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s <kmodel_path>\n", argv[0]);
    return 1;
  }
  g_kmodel_path = argv[1];

  struct sigaction sa = {};
  sa.sa_handler = HandleSignal;
  sigfillset(&sa.sa_mask);
  sigaction(SIGINT, &sa, nullptr);

  ee_benchmark_initialize();

  while (running_) {
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
