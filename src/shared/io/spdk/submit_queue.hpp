#pragma once

#include <atomic>
#include <stdexcept>
#include <string>
#include <vector>

namespace shared {
namespace spdk_submit_queue {

template <class T>
struct Queue {
  int head, tail, size, cap;
  T *data;

  explicit Queue(int cap) : head(0), tail(0), size(0), cap(cap) {
    if (cap <= 0) {
      throw std::invalid_argument("Queue capacity must be positive");
    }
    data = new T[cap];
  }

  ~Queue() { delete[] data; }

  bool empty() const { return size == 0; }

  void push(const T &x) {
    if (size == cap) {
      throw std::runtime_error("SPDK local queue full: capacity=" +
                               std::to_string(cap));
    }
    data[tail] = x;
    tail = (tail + 1) % cap;
    size++;
  }

  T pop() {
    if (empty()) {
      throw std::runtime_error("SPDK local queue pop on empty queue");
    }
    int r = head;
    head = (head + 1) % cap;
    size--;
    return data[r];
  }
};

template <class T>
struct SPSCQueue {
  std::atomic<int> head, tail;
  int cap;
  T *data;

  explicit SPSCQueue(int usable_capacity)
      : head(0), tail(0), cap(usable_capacity + 1) {
    if (usable_capacity <= 0) {
      throw std::invalid_argument("SPSCQueue capacity must be positive");
    }
    data = new T[cap];
  }

  ~SPSCQueue() { delete[] data; }

  void push(const T &x) {
    int t = tail.load();
    int next = (t + 1) % cap;
    if (next == head.load()) {
      throw std::runtime_error("SPDK submit queue full: capacity=" +
                               std::to_string(cap - 1));
    }
    data[t] = x;
    tail.store(next);
  }

  bool pop(T &x) {
    int h = head.load();
    int t = tail.load();
    if (h == t) {
      return false;
    }
    x = data[h];
    h = (h + 1) % cap;
    head.store(h);
    return true;
  }
};

template <class T>
struct MPSCQueue {
  int threads;
  int cur;
  int capacity;
  std::vector<SPSCQueue<T> *> q;

  MPSCQueue(int cap, int thread) : threads(thread), cur(0), capacity(cap) {
    if (thread <= 0) {
      throw std::invalid_argument("MPSCQueue thread count must be positive");
    }
    for (int i = 0; i < threads; i++) {
      q.push_back(new SPSCQueue<T>(cap));
    }
  }

  ~MPSCQueue() {
    for (auto *queue : q) {
      delete queue;
    }
  }

  void push(const T &x, int t) {
    if (t < 0 || t >= threads) {
      throw std::runtime_error("SPDK submit queue producer thread out of range: " +
                               std::to_string(t) + " not in [0," +
                               std::to_string(threads) + ")");
    }
    try {
      q[t]->push(x);
    } catch (const std::runtime_error &e) {
      throw std::runtime_error(std::string(e.what()) +
                               " producer_thread=" + std::to_string(t));
    }
  }

  bool pop(T &x) {
    for (int i = 0; i < threads; i++) {
      int t = cur;
      cur = (cur + 1) % threads;
      if (q[t]->pop(x)) {
        return true;
      }
    }
    return false;
  }
};

}  // namespace spdk_submit_queue
}  // namespace shared
