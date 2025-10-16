/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include <unistd.h>

#undef CEILING
#define CEILING(x, y) (((x) + (y)-1) / (y))

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            _exit(EXIT_FAILURE);                                            \
        }                                                                   \
    }

/*********************************************************************
 *
 *                   Device level utility functions
 *
 **********************************************************************/

// Get the SM id
__device__ __forceinline__ unsigned int get_smid(void) {
    unsigned int ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

// Get the warp id within the application
__device__ __forceinline__ unsigned int get_warpid(void) {
    unsigned int ret;
    asm("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

// Get the line id within the warp
__device__ __forceinline__ unsigned int get_laneid(void) {
    unsigned int laneid;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid));
    return laneid;
}

// Get a global warp id
__device__ __forceinline__ int get_global_warp_id() {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x +
                   gridDim.x * gridDim.y * blockIdx.z;

    int l_thread_id = (threadIdx.z * (blockDim.x * blockDim.y)) +
                      (threadIdx.y * blockDim.x) + threadIdx.x;

    int l_warp_id = l_thread_id / 32;

    int n_warps = CEILING(blockDim.x * blockDim.y * blockDim.z, 32);

    int g_warp_id = block_id * n_warps + l_warp_id;

    return g_warp_id;
}

// Get a thread's CTA ID
__device__ __forceinline__ int4 get_ctaid(void) {
    int4 ret;
    asm("mov.u32 %0, %ctaid.x;" : "=r"(ret.x));
    asm("mov.u32 %0, %ctaid.y;" : "=r"(ret.y));
    asm("mov.u32 %0, %ctaid.z;" : "=r"(ret.z));
    return ret;
}

//  Get the number of CTA ids per grid
__device__ __forceinline__ int4 get_nctaid(void) {
    int4 ret;
    asm("mov.u32 %0, %nctaid.x;" : "=r"(ret.x));
    asm("mov.u32 %0, %nctaid.y;" : "=r"(ret.y));
    asm("mov.u32 %0, %nctaid.z;" : "=r"(ret.z));
    return ret;
}

// Device level sleep function
__device__ __forceinline__ void csleep(uint64_t clock_count) {
    if (clock_count == 0) return;
    clock_t start_clock = clock64();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count) {
        clock_offset = clock64() - start_clock;
    }
}

class Managed {
  public:
    void *operator new(size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        return ptr;
    }

    // void Managed::operator delete(void *ptr)
    void operator delete(void *ptr) { cudaFree(ptr); }

    void *operator new[](size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        return ptr;
    }
    // void Managed::operator delete[] (void* ptr) {
    void operator delete[](void *ptr) { cudaFree(ptr); }
};
