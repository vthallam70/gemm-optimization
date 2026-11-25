// Single-precision GEMM kernels, row-major: C (MxN) = A (MxK) * B (KxN).
#pragma once
namespace gemm {
void naive(int M, int N, int K, const float* A, const float* B, float* C);      // i-j-k dot products
void reordered(int M, int N, int K, const float* A, const float* B, float* C);  // i-k-j: unit-stride inner loop
void blocked(int M, int N, int K, const float* A, const float* B, float* C);    // cache tiling on top of i-k-j
void simd(int M, int N, int K, const float* A, const float* B, float* C);       // packed panels + 6x16 AVX2/FMA micro-kernel
void threaded(int M, int N, int K, const float* A, const float* B, float* C, int threads);  // simd split across threads
}
