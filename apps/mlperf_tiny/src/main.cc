// Copyright 2024 The MLPerf Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
//
// K230 DUT entry point for MLPerf Tiny legacy UART harness.

#include <cstdio>

#include "api/internally_implemented.h"
#include "api/submitter_implemented.h"

extern const char* g_kmodel_path;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s <kmodel_path>\n", argv[0]);
    return 1;
  }
  g_kmodel_path = argv[1];

  ee_benchmark_initialize();

  while (true) {
    int c = th_getchar();
    // Filter CR/LF — RT-Smart msh delivers input line-buffered with
    // trailing CR.  The MLPerf protocol uses '%' as the sole terminator.
    if (c == '\r' || c == '\n') {
      continue;
    }
    ee_serial_callback(c);
  }

  return 0;
}
