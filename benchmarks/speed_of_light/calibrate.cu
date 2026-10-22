// Empirical ceilings: FP32 FMA issue, strict-FP32 cuBLAS, and DRAM triad.
// CUDA events time completed GPU work. No global clocks/power settings change.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while (0)
#define BLAS(x) do { cublasStatus_t e = (x); if (e != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %d at line %d\n", (int)e, __LINE__); exit(1); }} while (0)

__global__ void fill(float *a, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (size_t)blockDim.x * gridDim.x) {
    unsigned x = (unsigned)i * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    a[i] = ((x >> 22u) ^ x) * 0x1p-32f - 0.5f;
  }
}

__global__ void triad(float *a, const float *b, const float *c, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (size_t)blockDim.x * gridDim.x) a[i] = b[i] + 0.9f * c[i];
}

__global__ void fma_peak(float *output, int iterations, float alpha, float beta) {
  float a0 = threadIdx.x * 0.001f, a1 = a0 + .01f, a2 = a0 + .02f, a3 = a0 + .03f;
  float a4 = a0 + .04f, a5 = a0 + .05f, a6 = a0 + .06f, a7 = a0 + .07f;
  float a8 = a0 + .08f, a9 = a0 + .09f, a10 = a0 + .10f, a11 = a0 + .11f;
  float a12 = a0 + .12f, a13 = a0 + .13f, a14 = a0 + .14f, a15 = a0 + .15f;
  #pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    a0 = __fmaf_rn(a0, alpha, beta); a1 = __fmaf_rn(a1, alpha, beta);
    a2 = __fmaf_rn(a2, alpha, beta); a3 = __fmaf_rn(a3, alpha, beta);
    a4 = __fmaf_rn(a4, alpha, beta); a5 = __fmaf_rn(a5, alpha, beta);
    a6 = __fmaf_rn(a6, alpha, beta); a7 = __fmaf_rn(a7, alpha, beta);
    a8 = __fmaf_rn(a8, alpha, beta); a9 = __fmaf_rn(a9, alpha, beta);
    a10 = __fmaf_rn(a10, alpha, beta); a11 = __fmaf_rn(a11, alpha, beta);
    a12 = __fmaf_rn(a12, alpha, beta); a13 = __fmaf_rn(a13, alpha, beta);
    a14 = __fmaf_rn(a14, alpha, beta); a15 = __fmaf_rn(a15, alpha, beta);
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] =
    a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15;
}

template<class F> void measure(const char *name, F fn, double work, const char *unit) {
  cudaEvent_t start, end;
  CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
  // 300 untimed warmup launches; duration varies by calibration kernel.
  // No frequency locking or global power configuration changes.
  for (int i = 0; i < 300; ++i) fn();
  CUDA(cudaDeviceSynchronize());
  std::vector<float> times;
  for (int r = 0; r < 31; ++r) {
    CUDA(cudaEventRecord(start));
    for (int i = 0; i < 5; ++i) fn();
    CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
    float ms; CUDA(cudaEventElapsedTime(&ms, start, end)); times.push_back(ms / 5);
  }
  std::vector<float> sorted = times; std::sort(sorted.begin(), sorted.end());
  printf("{\"name\":\"%s\",\"median_ms\":%.9g,\"min_ms\":%.9g,\"max_ms\":%.9g,"
         "\"rate\":%.9g,\"unit\":\"%s\",\"work\":%.12g,\"samples_ms\":[",
         name, sorted[15], sorted.front(), sorted.back(), work / (sorted[15] * 1e-3), unit, work);
  for (size_t i = 0; i < times.size(); ++i) printf("%s%.9g", i ? "," : "", times[i]);
  printf("]}\n"); fflush(stdout);
  CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
}

int main() {
  cudaDeviceProp prop; CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("{\"device\":\"%s\",\"sm_count\":%d,\"cc_major\":%d,\"cc_minor\":%d,"
         "\"global_bytes\":%zu,\"l2_bytes\":%d,\"memory_bus_bits\":%d}\n",
         prop.name, prop.multiProcessorCount, prop.major, prop.minor,
         prop.totalGlobalMem, prop.l2CacheSize, prop.memoryBusWidth);
  const size_t n = 64 * 1024 * 1024; // 3 x 256 MiB, exceeds L2 by a wide margin.
  float *a, *b, *c;
  CUDA(cudaMalloc(&a, n * sizeof(float))); CUDA(cudaMalloc(&b, n * sizeof(float)));
  CUDA(cudaMalloc(&c, n * sizeof(float)));
  fill<<<1024, 256>>>(a, n); fill<<<1024, 256>>>(b, n); fill<<<1024, 256>>>(c, n);
  measure("dram_triad", [&] { triad<<<1024, 256>>>(a, b, c, n); },
          3.0 * n * sizeof(float), "bytes_per_second");
  int blocks = prop.multiProcessorCount * 16, threads = 256, iterations = 8192;
  measure("fp32_fma", [&] { fma_peak<<<blocks, threads>>>(a, iterations, .999f, .001f); },
          2.0 * 16 * iterations * blocks * threads, "flops_per_second");
  cublasHandle_t handle; BLAS(cublasCreate(&handle));
  BLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  const float alpha = 1.0f, beta = 0.0f;
  for (int d : {4096, 8192}) {
    char label[64]; snprintf(label, sizeof(label), "cublas_sgemm_%d", d);
    measure(label, [&] {
      BLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, d, d,
                      &alpha, b, d, c, d, &beta, a, d));
    }, 2.0 * d * d * d, "flops_per_second");
  }
  BLAS(cublasDestroy(handle));
  CUDA(cudaFree(a)); CUDA(cudaFree(b)); CUDA(cudaFree(c));
}
