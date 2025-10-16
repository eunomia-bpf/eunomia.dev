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

#include <inttypes.h>

namespace InstrType {

/* all supported arch have at most 255 general purpose registers */
constexpr const int RZ = 255;
/* the always true predicate is indicated as "7" on all the archs */
constexpr const int PT = 7;
/* the entire predicate register is encoded as "8" */
constexpr const int PR = 8;
constexpr const int URZ = 63;
constexpr const int UPT = 7;  // uniform predicate true
constexpr const int UPR = 8;  // entire uniform predicate register
constexpr const int MAX_CHARS = 256;

// loads and stores have 1, LDGSTS has 2
constexpr const int MAX_NUM_MREF_PER_INSTR = 2;

enum class MemorySpace {
    NONE,
    LOCAL,             // local memory operation
    GENERIC,           // generic memory operation
    GLOBAL,            // global memory operation
    SHARED,            // shared memory operation
    CONSTANT,          // constant memory operation
    GLOBAL_TO_SHARED,  // read from global memory then write to shared memory
    SURFACE,   // surface memory operation
    TEXTURE,   // texture memory operation
    DISTRIBUTED_SHARED,// distributed shared memory operation
    TENSOR_MEM,        // placeholder for tensor memory operation
    TENSOR_CORE_MEM,   // tensor core memory 
};
constexpr const char* MemorySpaceStr[] = {
    "NONE", "LOCAL", "GENERIC", "GLOBAL", "SHARED", "CONSTANT",
    "GLOBAL_TO_SHARED", "SURFACE", "TEXTURE",
    "DISTRIBUTED_SHARED", "TENSOR_MEM", "TENSOR_CORE_MEM"
};

enum class OperandType {
    IMM_UINT64,
    IMM_DOUBLE,
    REG,
    PRED,
    UREG,
    UPRED,
    CBANK,
    MREF,
    GENERIC,
    MEM_DESC
};

constexpr const char* OperandTypeStr[] = {
    "IMM_UINT64", "IMM_DOUBLE", "REG",  "PRED",   "UREG",
    "UPRED",      "CBANK",      "MREF", "GENERIC",
    "MEM_DESC"
};

enum class RegModifierType {
    NO_MOD = 0,
    /* stride modifiers */
    X1,
    X4,
    X8,
    X16,
    /* size modifiers */
    U32,
    U64,
};
constexpr const char* RegModifierTypeStr[] = {
    "NO_MOD", "X1", "X4", "X8", "X16", "U32", /* no U */ "64"};

typedef struct {
    /* operand type */
    OperandType type;
    /* operand string */
    char str[MAX_CHARS];
    /* is negative */
    bool is_neg;
    /* is not */
    bool is_not;
    /* is absolute */
    bool is_abs;
    /* operand size in byte */
    int nbytes;

    union {
        struct {
            uint64_t value;
        } imm_uint64;

        struct {
            double value;
        } imm_double;

        struct {
            int num;
            /* register properties .XXX */
            char prop[MAX_CHARS];
        } reg;

        struct {
            int num;
        } pred;

        struct {
            int id;
            bool has_imm_offset;
            int imm_offset;
            bool has_reg_offset;
            int reg_offset;
        } cbank;

        struct {
            bool has_ra;
            int ra_num;
            RegModifierType ra_mod;
            bool has_ur;
            int ur_num;
            RegModifierType ur_mod;
            bool has_imm;
            int imm;
            bool has_desc;
            int desc_ureg_num;
        } mref;

        struct {
            char array[MAX_CHARS];
        } generic;

        struct {
            int ureg_num;
        } mem_desc;
    } u;
} operand_t;
};
