// crash_handler.h
#pragma once
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <execinfo.h>
#include <unistd.h>

namespace crash {

[[gnu::noinline]] inline void print_backtrace_fd(int fd = STDERR_FILENO) {
  void *bt[128];
  int n = ::backtrace(bt, 128);
  ::dprintf(fd, "---- backtrace (n=%d) ----\n", n);
  ::backtrace_symbols_fd(bt, n, fd); // 需要 -rdynamic 才能符号化
  ::dprintf(fd, "---------------------------\n");
}

inline void signal_handler(int sig) {
  const char *name = strsignal(sig);
  ::dprintf(STDERR_FILENO, "\n*** FATAL SIGNAL %d (%s) ***\n", sig,
            name ? name : "?");
  print_backtrace_fd();
  // 避免在已损坏内存环境中做复杂清理，直接退出
  _Exit(128 + sig);
}

inline void install_handlers() {
  // 崩溃类信号
  ::signal(SIGSEGV, signal_handler);
  ::signal(SIGABRT, signal_handler);
  ::signal(SIGFPE, signal_handler);
  ::signal(SIGILL, signal_handler);
  ::signal(SIGBUS, signal_handler);

  // 捕获 std::terminate（例如异常逃逸）
  std::set_terminate([]() {
    ::dprintf(STDERR_FILENO, "\n*** std::terminate called ***\n");
    print_backtrace_fd();
    std::abort();
  });
}

} // namespace crash
