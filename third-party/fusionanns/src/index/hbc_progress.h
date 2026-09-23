#pragma once

#include <cstddef>
#include <iostream>
#include <string>

void HbcProgressStart(size_t total);
void HbcProgressStartLabeled(const std::string &label, size_t total);
void HbcProgressAdvance(size_t delta);
void HbcProgressFinish();
void HbcFlushProgressBeforeLog();

#ifdef _OPENMP
#define HBC_LOG(expr)                                                          \
  do {                                                                         \
    _Pragma("omp critical(hbc_log)") {                                         \
      HbcFlushProgressBeforeLog();                                             \
      std::cout << expr;                                                       \
    }                                                                          \
  } while (0)
#else
#define HBC_LOG(expr)                                                          \
  do {                                                                         \
    HbcFlushProgressBeforeLog();                                               \
    std::cout << expr;                                                         \
  } while (0)
#endif
