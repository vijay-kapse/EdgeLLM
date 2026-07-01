// INT8 GEMM microbenchmark: naive triple loop vs. SIMD-vectorized.
//
// Computes C[M,N] (int32) = A[M,K] (int8) * B^T[N,K] (int8), i.e. each output
// element is the dot product of an A row and a B row (B is stored row-major and
// pre-transposed so both operands are contiguous — the layout a real quantized
// matmul kernel would use). Reports GOPS and the SIMD speedup.
//
// SIMD path:
//   * ARM (Apple Silicon / Snapdragon): NEON. Uses the ARMv8.2 SDOT instruction
//     (vdotq_s32) when available, otherwise widening vmull_s8 + pairwise adds.
//   * x86: AVX2 (_mm256_maddubs / madd) when compiled with -mavx2.
//   * Otherwise: falls back to the scalar kernel (still correct).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define EDGELLM_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define EDGELLM_AVX2 1
#endif

namespace {

// Scalar baseline. Auto-vectorization is deliberately disabled so this is a
// genuine scalar reference for the hand-written SIMD kernel (otherwise -O3
// auto-vectorizes it and the comparison measures the compiler, not the kernel).
void naive_gemm(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t acc = 0;
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
      for (int k = 0; k < K; ++k) acc += static_cast<int32_t>(A[i * K + k]) * B[j * K + k];
      C[i * N + j] = acc;
    }
  }
}

#if defined(EDGELLM_NEON)
int32_t dot_neon(const int8_t* a, const int8_t* b, int K) {
  int k = 0;
#if defined(__ARM_FEATURE_DOTPROD)
  int32x4_t acc = vdupq_n_s32(0);
  for (; k + 16 <= K; k += 16) {
    int8x16_t va = vld1q_s8(a + k);
    int8x16_t vb = vld1q_s8(b + k);
    acc = vdotq_s32(acc, va, vb);  // 16 int8 MACs -> 4 int32 lanes
  }
  int32_t sum = vaddvq_s32(acc);
#else
  int32x4_t acc = vdupq_n_s32(0);
  for (; k + 16 <= K; k += 16) {
    int8x16_t va = vld1q_s8(a + k);
    int8x16_t vb = vld1q_s8(b + k);
    int16x8_t lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
    int16x8_t hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
    acc = vpadalq_s16(acc, lo);
    acc = vpadalq_s16(acc, hi);
  }
  int32_t sum = vaddvq_s32(acc);
#endif
  for (; k < K; ++k) sum += static_cast<int32_t>(a[k]) * b[k];
  return sum;
}
#elif defined(EDGELLM_AVX2)
int32_t dot_avx2(const int8_t* a, const int8_t* b, int K) {
  __m256i acc = _mm256_setzero_si256();
  int k = 0;
  for (; k + 32 <= K; k += 32) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + k));
    // multiply low/high halves as int16, then accumulate as int32.
    __m256i lo = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(va)),
                                   _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb)));
    __m256i hi = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1)),
                                   _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1)));
    acc = _mm256_add_epi32(acc, _mm256_add_epi32(lo, hi));
  }
  alignas(32) int32_t tmp[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc);
  int32_t sum = 0;
  for (int t = 0; t < 8; ++t) sum += tmp[t];
  for (; k < K; ++k) sum += static_cast<int32_t>(a[k]) * b[k];
  return sum;
}
#endif

void simd_gemm(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
#if defined(EDGELLM_NEON)
      C[i * N + j] = dot_neon(A + i * K, B + j * K, K);
#elif defined(EDGELLM_AVX2)
      C[i * N + j] = dot_avx2(A + i * K, B + j * K, K);
#else
      int32_t acc = 0;
      for (int k = 0; k < K; ++k) acc += static_cast<int32_t>(A[i * K + k]) * B[j * K + k];
      C[i * N + j] = acc;
#endif
    }
  }
}

double time_ms(void (*fn)(const int8_t*, const int8_t*, int32_t*, int, int, int),
               const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K, int iters) {
  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; ++it) fn(A, B, C, M, N, K);
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

}  // namespace

int main(int argc, char** argv) {
  int M = argc > 1 ? std::atoi(argv[1]) : 256;
  int N = argc > 2 ? std::atoi(argv[2]) : 256;
  int K = argc > 3 ? std::atoi(argv[3]) : 512;
  int iters = argc > 4 ? std::atoi(argv[4]) : 20;

#if defined(EDGELLM_NEON)
  const char* isa = "NEON";
#elif defined(EDGELLM_AVX2)
  const char* isa = "AVX2";
#else
  const char* isa = "scalar (no SIMD ISA detected)";
#endif

  std::vector<int8_t> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(N) * K);
  std::vector<int32_t> C0(static_cast<size_t>(M) * N), C1(static_cast<size_t>(M) * N);
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> dist(-127, 127);
  for (auto& x : A) x = static_cast<int8_t>(dist(rng));
  for (auto& x : B) x = static_cast<int8_t>(dist(rng));

  naive_gemm(A.data(), B.data(), C0.data(), M, N, K);
  simd_gemm(A.data(), B.data(), C1.data(), M, N, K);
  bool correct = C0 == C1;

  double naive_ms = time_ms(naive_gemm, A.data(), B.data(), C0.data(), M, N, K, iters);
  double simd_ms = time_ms(simd_gemm, A.data(), B.data(), C1.data(), M, N, K, iters);
  const double ops = 2.0 * M * N * K;  // one multiply + one add per MAC

  std::printf("INT8 GEMM  M=%d N=%d K=%d  iters=%d  SIMD=%s\n", M, N, K, iters, isa);
  std::printf("correctness (naive == simd): %s\n", correct ? "PASS" : "FAIL");
  std::printf("naive: %8.3f ms  (%6.2f GOPS)\n", naive_ms, ops / (naive_ms * 1e6));
  std::printf("simd : %8.3f ms  (%6.2f GOPS)\n", simd_ms, ops / (simd_ms * 1e6));
  std::printf("speedup: %.2fx\n", naive_ms / simd_ms);
  return correct ? 0 : 1;
}
