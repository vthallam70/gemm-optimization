#include "gemm.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <vector>

namespace gemm {

void naive(int M, int N, int K, const float* A, const float* B, float* C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];  // B walked down a column: stride N
      C[i * N + j] = s;
    }
}

void reordered(int M, int N, int K, const float* A, const float* B, float* C) {
  std::fill(C, C + static_cast<long>(M) * N, 0.f);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) {
      const float a = A[i * K + k];
      const float* b = B + static_cast<long>(k) * N;
      float* c = C + static_cast<long>(i) * N;
      for (int j = 0; j < N; ++j) c[j] += a * b[j];  // unit stride: the compiler can vectorize this
    }
}

void blocked(int M, int N, int K, const float* A, const float* B, float* C) {
  constexpr int BI = 64, BK = 256, BJ = 512;  // tile so the B block (256x512 floats = 512 KB) streams through L2/L3
  std::fill(C, C + static_cast<long>(M) * N, 0.f);
  for (int i0 = 0; i0 < M; i0 += BI)
    for (int k0 = 0; k0 < K; k0 += BK)
      for (int j0 = 0; j0 < N; j0 += BJ) {
        const int i1 = std::min(i0 + BI, M), k1 = std::min(k0 + BK, K), j1 = std::min(j0 + BJ, N);
        for (int i = i0; i < i1; ++i)
          for (int k = k0; k < k1; ++k) {
            const float a = A[i * K + k];
            const float* b = B + static_cast<long>(k) * N;
            float* c = C + static_cast<long>(i) * N;
            for (int j = j0; j < j1; ++j) c[j] += a * b[j];
          }
      }
}

// ---- packed AVX2/FMA kernel (BLIS-style loop nest) ----
namespace {
constexpr int MR = 6, NR = 16;               // micro-tile: 6 rows x 16 cols = 12 YMM accumulators
constexpr int KC = 256, MC = 120, NC = 2048;  // KC*NR panel of B stays in L1, MC*KC block of A in L2

float* alloc(long n) { return static_cast<float*>(std::aligned_alloc(64, ((n * sizeof(float) + 63) / 64) * 64)); }

// A block (mc x kc) -> panels of MR rows, k-major inside a panel, zero-padded rows.
void pack_A(int mc, int kc, const float* A, int lda, float* Ap) {
  for (int p = 0; p < mc; p += MR) {
    const int mr = std::min(MR, mc - p);
    for (int k = 0; k < kc; ++k)
      for (int r = 0; r < MR; ++r) *Ap++ = r < mr ? A[(p + r) * lda + k] : 0.f;
  }
}

// B block (kc x nc) -> panels of NR columns, k-major inside a panel, zero-padded columns.
void pack_B(int kc, int nc, const float* B, int ldb, float* Bp) {
  for (int q = 0; q < nc; q += NR) {
    const int nr = std::min(NR, nc - q);
    for (int k = 0; k < kc; ++k) {
      const float* b = B + static_cast<long>(k) * ldb + q;
      for (int c = 0; c < NR; ++c) *Bp++ = c < nr ? b[c] : 0.f;
    }
  }
}

// C[mr x nr] += Ap panel (MR x kc) * Bp panel (kc x NR)
inline void micro(int kc, const float* Ap, const float* Bp, float* C, int ldc, int mr, int nr) {
  __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00,
         c30 = c00, c31 = c00, c40 = c00, c41 = c00, c50 = c00, c51 = c00;
  for (int k = 0; k < kc; ++k, Ap += MR, Bp += NR) {
    const __m256 b0 = _mm256_load_ps(Bp), b1 = _mm256_load_ps(Bp + 8);
    __m256 a;
    a = _mm256_broadcast_ss(Ap + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
    a = _mm256_broadcast_ss(Ap + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
    a = _mm256_broadcast_ss(Ap + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
    a = _mm256_broadcast_ss(Ap + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
    a = _mm256_broadcast_ss(Ap + 4); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
    a = _mm256_broadcast_ss(Ap + 5); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
  }
  const __m256 acc[MR][2] = {{c00, c01}, {c10, c11}, {c20, c21}, {c30, c31}, {c40, c41}, {c50, c51}};
  if (mr == MR && nr == NR) {
    for (int r = 0; r < MR; ++r) {
      float* c = C + static_cast<long>(r) * ldc;
      _mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), acc[r][0]));
      _mm256_storeu_ps(c + 8, _mm256_add_ps(_mm256_loadu_ps(c + 8), acc[r][1]));
    }
  } else {  // edge tile: spill to a buffer and add only the valid part
    alignas(32) float t[MR * NR];
    for (int r = 0; r < MR; ++r) { _mm256_store_ps(t + r * NR, acc[r][0]); _mm256_store_ps(t + r * NR + 8, acc[r][1]); }
    for (int r = 0; r < mr; ++r)
      for (int c = 0; c < nr; ++c) C[static_cast<long>(r) * ldc + c] += t[r * NR + c];
  }
}

void simd_rows(int row0, int row1, int N, int K, const float* A, const float* B, float* C) {
  std::fill(C + static_cast<long>(row0) * N, C + static_cast<long>(row1) * N, 0.f);
  float* Ap = alloc(static_cast<long>(MC) * KC);
  float* Bp = alloc(static_cast<long>(KC) * NC);
  for (int jc = 0; jc < N; jc += NC) {
    const int nc = std::min(NC, N - jc);
    for (int pc = 0; pc < K; pc += KC) {
      const int kc = std::min(KC, K - pc);
      pack_B(kc, nc, B + static_cast<long>(pc) * N + jc, N, Bp);
      for (int ic = row0; ic < row1; ic += MC) {
        const int mc = std::min(MC, row1 - ic);
        pack_A(mc, kc, A + static_cast<long>(ic) * K + pc, K, Ap);
        for (int jr = 0; jr < nc; jr += NR)
          for (int ir = 0; ir < mc; ir += MR)
            micro(kc, Ap + static_cast<long>(ir) * kc, Bp + static_cast<long>(jr) * kc,
                  C + static_cast<long>(ic + ir) * N + jc + jr, N, std::min(MR, mc - ir), std::min(NR, nc - jr));
      }
    }
  }
  std::free(Ap); std::free(Bp);
}
}  // namespace

void simd(int M, int N, int K, const float* A, const float* B, float* C) { simd_rows(0, M, N, K, A, B, C); }

void threaded(int M, int N, int K, const float* A, const float* B, float* C, int threads) {
  threads = std::max(1, threads);
  const int per = ((M + threads - 1) / threads + MR - 1) / MR * MR;  // row chunks, multiple of MR
  std::vector<std::thread> pool;
  for (int r0 = 0; r0 < M; r0 += per) pool.emplace_back(simd_rows, r0, std::min(M, r0 + per), N, K, A, B, C);
  for (auto& t : pool) t.join();
}

}  // namespace gemm
