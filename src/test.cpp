// Checks every kernel against a double-precision reference on awkward shapes (edges, primes, tiny).
#include "gemm.hpp"
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

int main() {
  const int shapes[][3] = {{1, 1, 1}, {5, 7, 3}, {6, 16, 1}, {13, 17, 19}, {64, 64, 64}, {100, 257, 129},
                           {121, 33, 300}, {257, 2049, 260}, {300, 300, 513}};
  using Fn = std::function<void(int, int, int, const float*, const float*, float*)>;
  const std::pair<const char*, Fn> kernels[] = {
      {"naive", gemm::naive}, {"reordered", gemm::reordered}, {"blocked", gemm::blocked}, {"simd", gemm::simd},
      {"threaded", [](int M, int N, int K, const float* A, const float* B, float* C) { gemm::threaded(M, N, K, A, B, C, 8); }}};
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  int failures = 0, checks = 0;
  for (auto& s : shapes) {
    const int M = s[0], N = s[1], K = s[2];
    std::vector<float> A(M * K), B(K * N), C(M * N);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    std::vector<double> ref(M * N, 0.0);
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j) ref[i * N + j] += double(A[i * K + k]) * B[k * N + j];
    for (auto& [name, fn] : kernels) {
      std::fill(C.begin(), C.end(), 123.f);  // garbage: kernels must overwrite C, not accumulate into it
      fn(M, N, K, A.data(), B.data(), C.data());
      double worst = 0;
      for (int i = 0; i < M * N; ++i) worst = std::max(worst, std::abs(C[i] - ref[i]));
      const bool ok = worst <= 1e-5 * K + 1e-5;
      ++checks; failures += !ok;
      if (!ok) std::printf("FAIL %-9s %dx%dx%d max_abs_err=%.3g\n", name, M, N, K, worst);
    }
  }
  std::printf("%d/%d kernel-shape checks passed\n", checks - failures, checks);
  return failures != 0;
}
