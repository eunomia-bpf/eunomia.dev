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
#include <stdint.h>
#include <stdio.h>
#include <cassert>

#define SIGN_EXTEND64(x) ((((int64_t)(x)) << 32) >> 32)

/* generic address generation code */
extern "C" __device__ __noinline__ uint64_t
gen_mref_addr(uint32_t ra_high, int is_ra64, uint32_t ra_low, int ra_stride,
              uint32_t ru_high, int is_ru64, uint32_t ru_low, int32_t imm,
              uint32_t mref_idx /* unused */) {
    int64_t base_addr = 0;

    if (is_ra64) {
        base_addr +=
            (((uint64_t)ra_high) << 32) | ((uint64_t)ra_low * ra_stride);
    } else {
        base_addr += SIGN_EXTEND64(ra_low * ra_stride);
    }

    if (is_ru64) {
        base_addr += (((uint64_t)ru_high) << 32) | ((uint64_t)ru_low);
    } else {
        base_addr += SIGN_EXTEND64(ru_low);
    }

    uint64_t addr = base_addr + imm;
#if 0
    printf(
        "ra_high %d - is_ra64 %d - ra_low %d - ra_stride %d - ru_high %d - "
        "is_ru64 %d - ru_low %d - imm %d base_addr %lx addr %lx\n",
        ra_high, is_ra64, ra_low, ra_stride, ru_high, is_ru64, ru_low, imm,
        base_addr, addr);
#endif
    return addr;
}
__global__ void load_module_nvbit_kernel(int var) {
    printf("");
    if (var) {
        int tmp = gen_mref_addr(var, var, var, var, var, var, var, var, var);
        printf("%d\n", tmp);
    }
}
