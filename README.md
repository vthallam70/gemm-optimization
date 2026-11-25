# SGEMM optimization: from naive loops to 83% of vendor BLAS

Single-precision matrix multiply (`C = A * B`, row-major), the core operation of LLM training and inference, optimized step by step on a CPU, with a CUDA version for NVIDIA GPUs.

## CPU kernels (`src/`)

| Kernel | Idea |
|---|---|
| `naive` | i-j-k dot products; B is read down a column (stride N), so almost every access misses cache |
| `reordered` | i-k-j loop order: unit-stride inner loop the compiler can auto-vectorize |
| `blocked` | cache tiling on top of i-k-j |
| `simd` | BLIS-style: pack A (MC x KC) and B (KC x NC) into contiguous panels, then a hand-written 6x16 AVX2/FMA micro-kernel that keeps 12 YMM accumulators in registers (KC panel of B sized for L1, A block for L2) |
| `threaded` | `simd` with rows split across `std::thread` workers |

```sh
make            # clang++ -O3 -mavx2 -mfma
./test          # every kernel vs a double-precision reference on 9 awkward shapes (edge tiles, primes, 1x1x1)
./bench 8       # GFLOP/s per kernel and size vs Apple Accelerate cblas_sgemm
```

### Results (Nov 25, 2026; Intel Core i9-9880H, 8 cores, AVX2/FMA, macOS; median of 3–9 runs)

`./test`: 45/45 kernel-shape checks pass. Full data in `results_cpu.csv`.

| GFLOP/s | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| naive | 1.9 | 1.2 | 1.3 | – |
| reordered | 31.6 | 23.0 | 22.2 | 9.5 |
| blocked | 32.0 | 24.2 | 21.2 | 20.8 |
| simd (1 core) | 88.6 | 95.1 | 94.2 | 86.7 |
| threaded (8 threads) | 46.3 | 200.7 | 326.5 | 327.8 |
| Apple Accelerate | 136.3 | 384.6 | 499.5 | 393.0 |

- 1024: 247x faster than naive; 2048: 83% of Accelerate's throughput.
- Cache blocking matters once the working set leaves cache: at 2048, `blocked` is 2.2x `reordered`.
- At 256, threading loses to one core: each thread re-packs B and the work per thread is too small.
- Not yet done: prefetching, threading inside the packed loops instead of across rows (so B is packed once), and AVX-512.

## CUDA kernels (`cuda/`), not yet run

`gemm.cu` contains a coalesced naive kernel, 32x32 shared-memory tiling, and 2D register blocking (128x128 block tile, 8x8 per thread), each checked against cuBLAS (FP32, TF32 disabled) and timed with CUDA events. This machine has no NVIDIA GPU, so **this code has not been compiled or run yet**. Run it for free with `cuda/colab_gemm.ipynb` on a Colab T4, then record the results here.
