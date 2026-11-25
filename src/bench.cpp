// GFLOP/s for each kernel on square matrices; reference: the platform BLAS (Apple Accelerate cblas_sgemm).
#include "gemm.hpp"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <thread>
#include <vector>

static double median_seconds(const std::function<void()>& f, int reps) {
  f();  // warm-up
  std::vector<double> t;
  for (int r = 0; r < reps; ++r) {
    auto a = std::chrono::steady_clock::now(); f();
    t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main(int argc, char** argv) {
  const int threads = argc > 1 ? std::atoi(argv[1]) : int(std::thread::hardware_concurrency() / 2);
  std::printf("size,kernel,seconds,gflops\n");
  for (int n : {256, 512, 1024, 2048}) {
    std::vector<float> A(n * n), B(n * n), C(n * n);
    std::mt19937 rng(1); std::uniform_real_distribution<float> d(-1, 1);
    for (auto& x : A) x = d(rng);
    for (auto& x : B) x = d(rng);
    const double flops = 2.0 * n * n * n;
    const int reps = n <= 512 ? 9 : 5;
    auto run = [&](const char* name, std::function<void()> f, int r) {
      const double s = median_seconds(f, r);
      std::printf("%d,%s,%.6f,%.2f\n", n, name, s, flops / s / 1e9); std::fflush(stdout);
    };
    if (n <= 1024) run("naive", [&] { gemm::naive(n, n, n, A.data(), B.data(), C.data()); }, n <= 512 ? reps : 3);
    run("reordered", [&] { gemm::reordered(n, n, n, A.data(), B.data(), C.data()); }, reps);
    run("blocked", [&] { gemm::blocked(n, n, n, A.data(), B.data(), C.data()); }, reps);
    run("simd", [&] { gemm::simd(n, n, n, A.data(), B.data(), C.data()); }, reps);
    run("threaded", [&] { gemm::threaded(n, n, n, A.data(), B.data(), C.data(), threads); }, reps);
    run("accelerate_sgemm", [&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.f, A.data(), n, B.data(), n, 0.f, C.data(), n);
    }, reps);
  }
}
