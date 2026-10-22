#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
extern "C" {
void sol_start() { cudaProfilerStart(); }
void sol_stop() { cudaProfilerStop(); }
void sol_push(const char *name) { nvtxRangePushA(name); }
void sol_pop() { nvtxRangePop(); }
}
