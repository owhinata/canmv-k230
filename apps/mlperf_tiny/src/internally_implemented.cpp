// Copied from mlperf_tiny/benchmark/api/internally_implemented.cpp
// Modified: TH_MODEL_VERSION replaced with runtime variable model_version_.
//
// Copyright 2020 EEMBC and The MLPerf Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include "api/internally_implemented.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "api/submitter_implemented.h"

// Runtime model version — set by main() from argv or kmodel filename.
extern const char* model_version_;

// Command buffer (incoming commands from host)
char volatile g_cmd_buf[EE_CMD_SIZE + 1];
size_t volatile g_cmd_pos = 0u;

// Generic buffer to db input.
uint8_t gp_buff[MAX_DB_INPUT_SIZE];
size_t g_buff_size = 0u;
size_t g_buff_pos = 0u;

bool g_state_parser_enabled = false;

void ee_serial_callback(char c) {
  if (c == EE_CMD_TERMINATOR) {
    g_cmd_buf[g_cmd_pos] = (char)0;
    th_command_ready(g_cmd_buf);
    g_cmd_pos = 0;
  } else {
    g_cmd_buf[g_cmd_pos] = c;
    g_cmd_pos = g_cmd_pos >= EE_CMD_SIZE ? EE_CMD_SIZE : g_cmd_pos + 1;
  }
}

void ee_serial_command_parser_callback(char *p_command) {
  char *tok;

  if (g_state_parser_enabled != true) {
    return;
  }

  tok = strtok(p_command, EE_CMD_DELIMITER);

  if (strncmp(tok, EE_CMD_NAME, EE_CMD_SIZE) == 0) {
    th_printf(EE_MSG_NAME, EE_DEVICE_NAME, TH_VENDOR_NAME_STRING);
  } else if (strncmp(tok, EE_CMD_TIMESTAMP, EE_CMD_SIZE) == 0) {
    th_timestamp();
  } else if (ee_profile_parse(tok) == EE_ARG_CLAIMED) {
  } else {
    th_printf(EE_ERR_CMD, tok);
  }

  th_printf(EE_MSG_READY);
}

void ee_benchmark_initialize(void) {
  th_serialport_initialize();
  th_timestamp_initialize();
  th_final_initialize();
  th_printf(EE_MSG_INIT_DONE);
  g_state_parser_enabled = true;
  th_printf(EE_MSG_READY);
}

arg_claimed_t ee_profile_parse(char *command) {
  char *p_next;

  if (strncmp(command, "profile", EE_CMD_SIZE) == 0) {
    th_printf("m-profile-[%s]\r\n", EE_FW_VERSION);
    th_printf("m-model-[%s]\r\n", model_version_);
  } else if (strncmp(command, "help", EE_CMD_SIZE) == 0) {
    th_printf("%s\r\n", EE_FW_VERSION);
    th_printf("\r\n");
    th_printf("help         : Print this information\r\n");
    th_printf("name         : Print the name of the device\r\n");
    th_printf("timestsamp   : Generate a timetsamp\r\n");
    th_printf("db SUBCMD    : Manipulate a generic byte buffer\r\n");
    th_printf("  load N     : Allocate N bytes and set load counter\r\n");
    th_printf("  db HH[HH]* : Load 8-bit hex byte(s) until N bytes\r\n");
    th_printf("  print [N=16] [offset=0]\r\n");
    th_printf("             : Print N bytes at offset as hex\r\n");
    th_printf(
        "infer N [W=0]: Load input, execute N inferences after W warmup "
        "loops\r\n");
    th_printf("results      : Return the result fp32 vector\r\n");
  } else if (ee_buffer_parse(command) == EE_ARG_CLAIMED) {
  } else if (strncmp(command, "infer", EE_CMD_SIZE) == 0) {
    size_t n = 1;
    size_t w = 10;
    int i;

    p_next = strtok(NULL, EE_CMD_DELIMITER);
    if (p_next) {
      i = atoi(p_next);
      if (i <= 0) {
        th_printf("e-[Inference iterations must be >0]\r\n");
        return EE_ARG_CLAIMED;
      }
      n = (size_t)i;
      p_next = strtok(NULL, EE_CMD_DELIMITER);
      if (p_next) {
        i = atoi(p_next);
        if (i < 0) {
          th_printf("e-[Inference warmup must be >=0]\r\n");
          return EE_ARG_CLAIMED;
        }
        w = (size_t)i;
      }
    }

    ee_infer(n, w);
  } else if (strncmp(command, "results", EE_CMD_SIZE) == 0) {
    th_results();
  } else {
    return EE_ARG_UNCLAIMED;
  }
  return EE_ARG_CLAIMED;
}

void ee_infer(size_t n, size_t n_warmup) {
  th_load_tensor();
  th_printf("m-warmup-start-%d\r\n", n_warmup);
  while (n_warmup-- > 0) {
    th_infer();
  }
  th_printf("m-warmup-done\r\n");
  th_printf("m-infer-start-%d\r\n", n);
  th_timestamp();
  th_pre();
  while (n-- > 0) {
    th_infer();
  }
  th_post();
  th_timestamp();
  th_printf("m-infer-done\r\n");
  th_results();
}

arg_claimed_t ee_buffer_parse(char *p_command) {
  char *p_next;

  if (strncmp(p_command, "db", EE_CMD_SIZE) != 0) {
    return EE_ARG_UNCLAIMED;
  }

  p_next = strtok(NULL, EE_CMD_DELIMITER);

  if (p_next == NULL) {
    th_printf("e-[Command 'db' requires a subcommand]\r\n");
  } else if (strncmp(p_next, "load", EE_CMD_SIZE) == 0) {
    p_next = strtok(NULL, EE_CMD_DELIMITER);

    if (p_next == NULL) {
      th_printf("e-[Command 'db load' requires the # of bytes]\r\n");
    } else {
      g_buff_size = (size_t)atoi(p_next);
      if (g_buff_size == 0) {
        th_printf("e-[Command 'db load' must be >0 bytes]\r\n");
      } else {
        g_buff_pos = 0;
        if (g_buff_size > MAX_DB_INPUT_SIZE) {
          th_printf("Supplied buffer size %d exceeds maximum of %d\n",
                    g_buff_size, MAX_DB_INPUT_SIZE);
        } else {
          th_printf("m-[Expecting %d bytes]\r\n", g_buff_size);
        }
      }
    }
  } else if (strncmp(p_next, "print", EE_CMD_SIZE) == 0) {
    size_t i = 0;
    const size_t max = 8;
    for (; i < g_buff_size; ++i) {
    if ((i + max) % max == 0 || i == 0) {
        th_printf("m-buffer-");
    }
    th_printf("%02x", gp_buff[i]);
    if (((i + 1) % max == 0) || ((i + 1) == g_buff_size)) {
        th_printf("\r\n");
    } else {
        th_printf("-");
    }
    }
    if (i % max != 0) {
    th_printf("\r\n");
    }
  } else {
    size_t numbytes;
    char test[3];
    long res;

    numbytes = th_strnlen(p_next, EE_CMD_SIZE);

    if ((numbytes & 1) != 0) {
      th_printf("e-[Insufficent number of hex digits]\r\n");
      return EE_ARG_CLAIMED;
    }
    test[2] = 0;
    for (size_t i = 0; i < numbytes;) {
      test[0] = p_next[i++];
      test[1] = p_next[i++];
      res = ee_hexdec(test);
      if (res < 0) {
        th_printf("e-[Invalid hex digit '%s']\r\n", test);
        return EE_ARG_CLAIMED;
      } else {
        gp_buff[g_buff_pos] = (uint8_t)res;
        g_buff_pos++;
        if (g_buff_pos == g_buff_size) {
          th_printf("m-load-done\r\n");
          return EE_ARG_CLAIMED;
        }
      }
    }
  }
  return EE_ARG_CLAIMED;
}

long ee_hexdec(char *hex) {
  char c;
  long dec = 0;
  long ret = 0;

  while (*hex && ret >= 0) {
    c = *hex++;
    if (c >= '0' && c <= '9') {
      dec = c - '0';
    } else if (c >= 'a' && c <= 'f') {
      dec = c - 'a' + 10;
    } else if (c >= 'A' && c <= 'F') {
      dec = c - 'A' + 10;
    } else {
      return -1;
    }
    ret = (ret << 4) + dec;
  }
  return ret;
}

size_t ee_get_buffer(uint8_t* buffer, size_t max_len) {
  int len = max_len < g_buff_pos ? max_len : g_buff_pos;
  if (buffer != nullptr) {
    memcpy(buffer, gp_buff, len * sizeof(uint8_t));
  }
  return len;
}
