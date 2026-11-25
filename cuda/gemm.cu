// CUDA SGEMM kernels (row-major C = A * B), checked and timed against cuBLAS.
// Build: nvcc -O3 -arch=native gemm.cu -lcublas -o gemm   Run: ./gemm
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

// 1) one thread per output, threadIdx.x along columns so B and C accesses coalesce
__global__ void naive(int M, int N, int K, const float* A, const float* B, float* C) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float s = 0.f;
  for (int k = 0; k < K; ++k) s += A[row * K + k] * B[k * N + col];
  C[row * N + col] = s;
}

// 2) 32x32 shared-memory tiles: each global element is loaded once per tile instead of once per thread
constexpr int T = 32;
__global__ void smem_tiled(int M, int N, int K, const float* A, const float* B, float* C) {
  __shared__ float As[T][T], Bs[T][T];
  const int tx = threadIdx.x, ty = threadIdx.y, row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
  float s = 0.f;
  for (int t = 0; t < K; t += T) {
    As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.f;
    Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < T; ++k) s += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = s;
}

// 3) 2D register blocking: 128x128 block tile, each of 256 threads computes an 8x8 sub-tile in registers.
//    Requires M, N multiples of 128 and K a multiple of 8.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN)) reg_tiled(int M, int N, int K, const float* A, const float* B, float* C) {
  constexpr int NT = (BM * BN) / (TM * TN);
  __shared__ float As[BM * BK], Bs[BK * BN];
  const int threadCol = threadIdx.x % (BN / TN), threadRow = threadIdx.x / (BN / TN);
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;
  const int rA = threadIdx.x / BK, cA = threadIdx.x % BK, rB = threadIdx.x / BN, cB = threadIdx.x % BN;
  float acc[TM * TN] = {0.f}, regM[TM], regN[TN];
  for (int bk = 0; bk < K; bk += BK) {
    for (int o = 0; o < BM; o += NT / BK) As[(rA + o) * BK + cA] = A[(rA + o) * K + cA];
    for (int o = 0; o < BK; o += NT / BN) Bs[(rB + o) * BN + cB] = B[(rB + o) * N + cB];
    __syncthreads();
    A += BK; B += BK * N;
#pragma unroll
    for (int d = 0; d < BK; ++d) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + d];
#pragma unroll
      for (int i = 0; i < TN; ++i) regN[i] = Bs[d * BN + threadCol * TN + i];
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) acc[m * TN + n] += regM[m] * regN[n];
    }
    __syncthreads();
  }
  for (int m = 0; m < TM; ++m)
    for (int n = 0; n < TN; ++n) C[(threadRow * TM + m) * N + threadCol * TN + n] = acc[m * TN + n];
}

template <class F> float median_ms(F f, int reps) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  f(); CK(cudaDeviceSynchronize());  // warm-up
  std::vector<float> t;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms);
  }
  CK(cudaGetLastError());
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  std::printf("# GPU: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
  std::printf("size,kernel,ms,gflops,max_rel_err_vs_cublas\n");
  cublasHandle_t h; cublasCreate(&h);
  cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH);  // plain FP32, no TF32, for a like-for-like comparison
  for (int n : {1024, 2048, 4096}) {
    const size_t bytes = size_t(n) * n * sizeof(float);
    std::vector<float> hA(size_t(n) * n), hB(size_t(n) * n), hRef(size_t(n) * n), hC(size_t(n) * n);
    std::mt19937 rng(1); std::uniform_real_distribution<float> d(-1, 1);
    for (auto& x : hA) x = d(rng);
    for (auto& x : hB) x = d(rng);
    float *A, *B, *C, *R;
    CK(cudaMalloc(&A, bytes)); CK(cudaMalloc(&B, bytes)); CK(cudaMalloc(&C, bytes)); CK(cudaMalloc(&R, bytes));
    CK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice)); CK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));
    const float one = 1.f, zero = 0.f;
    // row-major C = A*B is column-major C^T = B^T * A^T
    auto cublas = [&](float* out) { cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, B, n, A, n, &zero, out, n); };
    const double flops = 2.0 * n * n * n;
    const int reps = n <= 2048 ? 10 : 5;
    auto report = [&](const char* name, float ms, bool check) {
      double err = 0;
      if (check) {
        CK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < hC.size(); ++i) err = std::max(err, std::abs(double(hC[i]) - hRef[i]) / (std::abs(double(hRef[i])) + 1.0));
      }
      std::printf("%d,%s,%.3f,%.1f,%.2e\n", n, name, ms, flops / (ms * 1e-3) / 1e9, err); std::fflush(stdout);
    };
    report("cublas_sgemm", median_ms([&] { cublas(R); }, reps), false);
    CK(cudaMemcpy(hRef.data(), R, bytes, cudaMemcpyDeviceToHost));
    dim3 b16(16, 16), gN((n + 15) / 16, (n + 15) / 16);
    if (n <= 2048) report("naive", median_ms([&] { naive<<<gN, b16>>>(n, n, n, A, B, C); }, reps), true);
    dim3 bT(T, T), gT((n + T - 1) / T, (n + T - 1) / T);
    report("smem_tiled", median_ms([&] { smem_tiled<<<gT, bT>>>(n, n, n, A, B, C); }, reps), true);
    dim3 gR(n / 128, n / 128);
    report("reg_tiled", median_ms([&] { reg_tiled<128, 128, 8, 8, 8><<<gR, 256>>>(n, n, n, A, B, C); }, reps), true);
    CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C)); CK(cudaFree(R));
  }
  cublasDestroy(h);
}
