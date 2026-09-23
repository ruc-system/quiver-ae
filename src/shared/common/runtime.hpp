#pragma once

#include <cstdio>
#include <cstdlib>

#include <pthread.h>
#include <sched.h>
#include <sys/time.h>

namespace shared {

inline double elapsed() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

inline void bind_core(int core_num) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core_num, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set) != 0) {
    perror("pthread_setaffinity_np");
    std::exit(-1);
  }
}

}  // namespace shared
