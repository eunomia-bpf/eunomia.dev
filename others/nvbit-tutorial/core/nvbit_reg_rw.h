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

/* This file needs to be include once in your nvbit tool, it provides hooks to
 * the nvbit core library to properly load this tool.
 * Do not modify!!!  */
#pragma once
#include <stdio.h>
#include <cassert>
#include <stdint.h>

volatile __device__ int32_t __nvbit_var = 0;

#define FORCE_LINKING(x) { void* volatile dummy = &x; }
#define TAKE_CODE_SPACE(s) \
    { _Pragma("unroll") \
      for (int i = 0; i < s; i++) __nvbit_var += i;\
    }

/* parameters need to be used in the function to prevent compiler optimizing
 * them away. */

extern "C" __device__ __noinline__ int32_t nvbit_read_reg(uint64_t reg_num) {
    TAKE_CODE_SPACE(1024)
    FORCE_LINKING(reg_num);
    return __nvbit_var;
}

extern "C" __device__ __noinline__ void nvbit_write_reg(uint64_t reg_num,
                                                        int32_t reg_val) {
    TAKE_CODE_SPACE(1024)
    FORCE_LINKING(reg_num);
    FORCE_LINKING(reg_val);
}

extern "C" __device__ __noinline__ int32_t nvbit_read_ureg(uint64_t reg_num) {
    TAKE_CODE_SPACE(512)
    FORCE_LINKING(reg_num);
    return __nvbit_var;
}

extern "C" __device__ __noinline__ void nvbit_write_ureg(uint64_t reg_num,
                                                        int32_t reg_val) {
    TAKE_CODE_SPACE(512)
    FORCE_LINKING(reg_num);
    FORCE_LINKING(reg_val);
    // assert(__nvbit_var == reg_num + reg_val);
}

extern "C" __device__ __noinline__ int32_t nvbit_read_pred_reg() {
    TAKE_CODE_SPACE(32)
    return __nvbit_var;
}

extern "C" __device__ __noinline__ void nvbit_write_pred_reg(int32_t reg_val) {
    TAKE_CODE_SPACE(32)
    FORCE_LINKING(reg_val);
}

extern "C" __device__ __noinline__ int32_t nvbit_read_upred_reg() {
    TAKE_CODE_SPACE(32)
    return __nvbit_var;
}

extern "C" __device__ __noinline__ void nvbit_write_upred_reg(int32_t reg_val) {
    TAKE_CODE_SPACE(32)
    FORCE_LINKING(reg_val);
}
