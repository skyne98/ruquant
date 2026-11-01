// gfx906 instantiations of the skinny W4A4 kernels, compiled to asm by build.sh and read by
// isa.py. Not linked into anything: there is no MI50 on this machine.
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)
#include "w4a4_skinny.h"

#define RQ_INST(T) \
    template __global__ void w4a4_skinny<T>(const int32_t*, const int32_t*, int32_t*, int, int, int);
RQ_INST(1)
RQ_INST(2)
RQ_INST(4)
RQ_INST(8)
RQ_INST(16)
RQ_INST(24)
RQ_INST(32)
RQ_INST(48)
RQ_INST(64)
