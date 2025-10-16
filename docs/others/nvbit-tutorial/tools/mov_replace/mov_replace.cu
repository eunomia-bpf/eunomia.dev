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

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unordered_set>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
void nvbit_at_init() {
    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        /* Get the vector of instruction composing the loaded CUFunction "func"
         */
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, func);

        /* If verbose we print function name and number of" static" instructions
         */
        if (verbose) {
            printf("inspecting %s - num instrs %ld\n",
                   nvbit_get_func_name(ctx, func), instrs.size());
        }

        /* We iterate on the vector of instruction */
        for (auto instr : instrs) {
            /* Check if the instruction falls in the interval where we want to
             * instrument */
            if (instr->getIdx() < instr_begin_interval ||
                instr->getIdx() >= instr_end_interval) {
                continue;
            }

            std::string opcode = instr->getOpcode();
            /* match every MOV instruction */
            if (opcode.compare(0, 3, "MOV") == 0) {
               /* assert MOV has really two arguments */
                assert(instr->getNumOperands() == 2);
                const InstrType::operand_t *op0 = instr->getOperand(0);
                assert(op0->type == InstrType::OperandType::REG);
                const InstrType::operand_t *op1 = instr->getOperand(1);

                if(op1->type !=  InstrType::OperandType::REG &&
                   op1->type != InstrType::OperandType::UREG &&
                   op1->type != InstrType::OperandType::IMM_UINT64 &&
                   op1->type != InstrType::OperandType::CBANK) {
                    instr->printDecoded();
                    printf("Operand %s not handled\n", InstrType::OperandTypeStr[(int) op1->type]);
                    continue;
                }

                if (verbose) {
                    instr->printDecoded();
                }

                /* Insert a call to "mov_replace" before the instruction */
                nvbit_insert_call(instr, "mov_replace", IPOINT_BEFORE);

                /* Add predicate as argument to the instrumentation function */
                nvbit_add_call_arg_guard_pred_val(instr);

                /* Add destination register number as argument (first operand
                 * must be a register)*/
                nvbit_add_call_arg_const_val32(instr, op0->u.reg.num);

                /* add second operand */

                /* 0: non reg, 1: vector reg, 2: uniform reg */
                int is_op1_reg = 0;
                if (op1->type == InstrType::OperandType::REG) {
                    is_op1_reg = 1;
                    /* register number as immediate */
                    nvbit_add_call_arg_const_val32(instr, op1->u.reg.num);

                } else if (op1->type == InstrType::OperandType::UREG) {
                    is_op1_reg = 2;
                    /* register number as immediate */
                    nvbit_add_call_arg_const_val32(instr, op1->u.reg.num);

                } else if (op1->type == InstrType::OperandType::IMM_UINT64) {
                    /* Add immediate value (registers are 32 bits so immediate
                     * is also 32-bit and we can cast to int safely) */
                    nvbit_add_call_arg_const_val32(
                        instr, (int)op1->u.imm_uint64.value);

                } else if (op1->type == InstrType::OperandType::CBANK) {
                    /* Add value from constant bank (passed as immediate to
                     * the mov_replace function) */
                    nvbit_add_call_arg_cbank_val(instr, op1->u.cbank.id,
                                                 op1->u.cbank.imm_offset);
                }

                /* Add flag to specify if value or register number */
                nvbit_add_call_arg_const_val32(instr, is_op1_reg);

                /* Remove original instruction */
                nvbit_remove_orig(instr);
            }
        }
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        /* cast params to launch parameter based on cbid since if we are here
         * we know these are the right parameters types */
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
        }

        if (!is_exit) {
            instrument_function_if_needed(ctx, func);
            nvbit_enable_instrumented(ctx, func, true);
        }
    }
}

