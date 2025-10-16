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
#include <unordered_map>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

#define MAX_NUM_KERNEL 100

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
uint64_t tot_app_instrs = 0;

struct kernel_info {
    uint32_t kernel_id;
};
/* per kernel instruction counter, updated by the GPU */
std::unordered_map<CUfunction, kernel_info> kernel_map;
__managed__ uint64_t kernel_counter[MAX_NUM_KERNEL];

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
uint32_t start_grid_num = 0;
uint32_t end_grid_num = UINT32_MAX;
int verbose = 0;
int count_warp_level = 1;
int exclude_pred_off = 0;
int active_from_start = 1;
bool mangled = false;

/* used to select region of insterest when active from start is off */
bool active_region = true;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
pthread_mutex_t mutex;

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
void nvbit_at_init() {
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(start_grid_num, "START_GRID_NUM", 0,
                "Beginning of the kernel gird launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(
        end_grid_num, "END_GRID_NUM", UINT32_MAX,
        "End of the kernel launch interval where to apply instrumentation");
    GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 1,
                "Count warp level or thread level instructions");
    GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 0,
                "Exclude predicated off instruction from count");
    GET_VAR_INT(
        active_from_start, "ACTIVE_FROM_START", 1,
        "Start instruction counting from start or wait for cuProfilerStart "
        "and cuProfilerStop");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1,
                "Print kernel names mangled or not");

    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    if (active_from_start == 0) {
        active_region = false;
    }

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

        /* Get the vector of instruction composing the loaded CUFunction "f" */
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        /* If verbose we print function name and number of" static" instructions
         */
        if (verbose) {
            printf("inspecting %s - num instrs %ld\n",
                   nvbit_get_func_name(ctx, f), instrs.size());
        }

        /* We iterate on the vector of instruction */
        for (auto i : instrs) {
            /* Check if the instruction falls in the interval where we want to
             * instrument */
            if (i->getIdx() >= instr_begin_interval &&
                i->getIdx() < instr_end_interval) {
                /* If verbose we print which instruction we are instrumenting
                 * (both offset in the function and SASS string) */
                if (verbose == 1) {
                    i->print();
                } else if (verbose == 2) {
                    i->printDecoded();
                }

                /* Insert a call to "count_instrs" before the instruction "i" */
                nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
                if (exclude_pred_off) {
                    /* pass predicate value */
                    nvbit_add_call_arg_guard_pred_val(i);
                } else {
                    /* pass always true */
                    nvbit_add_call_arg_const_val32(i, 1);
                }

                /* add count warps option */
                nvbit_add_call_arg_const_val32(i, count_warp_level);
                /* add pointer to counter location */
                nvbit_add_call_arg_const_val64(i, (uint64_t)&kernel_counter[kernel_map[f].kernel_id]);
            }
        }
    }
}

void try_to_instrument(CUfunction f, CUcontext ctx) {
    /* if we are entering in a kernel launch:
     * 1. Lock the mutex to prevent multiple kernels to run concurrently
     * (overriding the counter) in case the user application does that
     * 2. Instrument the function if needed
     * 3. Select if we want to run the instrumented or original
     * version of the kernel
     * 4. Reset the kernel instruction counter */

    /* skip encountered kernels */
    if (kernel_map.find(f) != kernel_map.end()) {
        return;
    }
    /* stop instrumenting kernels if we run out of kernel_counters */
    if (kernel_id >= MAX_NUM_KERNEL) {
        /* keep record of total number of launched kernels */
        kernel_id++;
        return;
    }
    kernel_map[f].kernel_id = kernel_id++;
    instrument_function_if_needed(ctx, f);

    if (active_from_start) {
        if (kernel_map[f].kernel_id >= start_grid_num &&
            kernel_map[f].kernel_id < end_grid_num) {
            active_region = true;
        } else {
            active_region = false;
        }
    }

    if (active_region) {
        nvbit_enable_instrumented(ctx, f, true);
    } else {
        nvbit_enable_instrumented(ctx, f, false);
    }
}

void print_kernel_stats(CUfunction f, CUcontext ctx, int num_ctas) {
    tot_app_instrs += kernel_counter[kernel_map[f].kernel_id];
    printf(
        "\nkernel %d - %s - #thread-blocks %d,  kernel "
        "instructions %ld, total instructions %ld\n",
        kernel_map[f].kernel_id, nvbit_get_func_name(ctx, f, mangled), num_ctas,
        kernel_counter[kernel_map[f].kernel_id], tot_app_instrs);

    kernel_counter[kernel_map[f].kernel_id] = 0;
}

/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids  which are defined in tools_cuda_api_meta.h.
 * This call back is triggered bith at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /*
     * To support kernel launched by cuda graph (in addition to existing kernel
     * launche method), we need to do:
     *
     * 1. instrument kernels at cudaGraphAddKernelNode event. This is for cases
     * that kernels are manually added to a cuda graph.
     * 2. distinguish captured kernels when kernels are recorded to a graph
     * using stream capture. cudaStreamIsCapturing() tells us whether a stream
     * is capturiong.
     * 3. per-kernel instruction counters, since cuda graph can launch multiple
     * kernels at the same time.
     *
     * Three cases:
     *
     * 1. original kernel launch:
     *     1a. for any kernel launch without using a stream, we instrument it
     *     before it is launched, call cudaDeviceSynchronize after it is
     *     launched and read the instruction counter of the kernel.
     *     1b. for any kernel launch using a stream, but the stream is not
     *     capturing, we do the same thing as 1a.
     *
     *  2. cuda graph using stream capturing: if a kernel is launched in a
     *  stream and the stream is capturing. We instrument the kernel before it
     *  is launched and do nothing after it is launched, because the kernel is
     *  not running until cudaGraphLaunch. Instead, we issue a
     *  cudaStreamSynchronize after cudaGraphLaunch is done and reset the
     *  instruction counters, since a cloned graph might be launched afterwards.
     *
     *  3. cuda graph manual: we instrument the kernel added by
     *  cudaGraphAddKernelNode and do the same thing for cudaGraphLaunch as 2.
     *
     *  The above method should handle most of cuda graph launch cases.
     */
    /* kernel launches with stream parameter, they can be used for cuda graph */
    if (cbid == API_CUDA_cuLaunchKernel_ptsz  || cbid == API_CUDA_cuLaunchKernel
        || cbid == API_CUDA_cuLaunchGridAsync || cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        CUfunction f;
        CUstream hStream;

        if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {
            cuLaunchKernel_ptsz_params *p = (cuLaunchKernel_ptsz_params *)params;
            f = p->f;
            hStream = p->hStream;
        } else if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            f = p->f;
            hStream = p->config->hStream;
        } else {
            cuLaunchGridAsync_params *p = (cuLaunchGridAsync_params *)params;
            f = p->f;
            hStream = p->hStream;
        }

        if (!is_exit) {
            pthread_mutex_lock(&mutex);
            try_to_instrument(f, ctx);
        } else {
            /* check if the stream is capturing, if yes, do not sync */
            cudaStreamCaptureStatus streamStatus;
            CUDA_SAFECALL(cudaStreamIsCapturing(hStream, &streamStatus));

            if (streamStatus != cudaStreamCaptureStatusActive) {
                int num_ctas = 0;

                if (verbose >= 1) {
                    printf("kernel %s not captured by cuda graph\n", nvbit_get_func_name(ctx, f, mangled));
                }
                /* keep the old behavior (sync after each kernel is completed)
                 * if cuda graph is not used. */
                /* if we are exiting a kernel launch:
                 * 1. Wait until the kernel is completed using
                 * cudaDeviceSynchronize()
                 * 2. Get number of thread blocks in the kernel
                 * 3. Print the thread instruction counters
                 * 4. Release the lock*/
                CUDA_SAFECALL(cudaDeviceSynchronize());

                if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                    cbid == API_CUDA_cuLaunchKernel) {
                    cuLaunchKernel_params *p2 = (cuLaunchKernel_params *)params;
                    num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
                } else if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                    cbid == API_CUDA_cuLaunchKernelEx) {
                    cuLaunchKernelEx_params *p2 = (cuLaunchKernelEx_params *)params;
                    num_ctas = p2->config->gridDimX * p2->config->gridDimY *
                        p2->config->gridDimZ;
                }
                print_kernel_stats(f, ctx, num_ctas);
            } else {
                if (verbose >= 1) {
                    printf("kernel %s captured by cuda graph\n", nvbit_get_func_name(ctx, f, mangled));
                }
            }
            pthread_mutex_unlock(&mutex);
        }
    }
    if (cbid == API_CUDA_cuGraphAddKernelNode) {
        cuGraphAddKernelNode_params *p = (cuGraphAddKernelNode_params *)params;
        CUfunction f = p->nodeParams->func;

        if (!is_exit) {
            pthread_mutex_lock(&mutex);
            try_to_instrument(f, ctx);
        } else {
            pthread_mutex_unlock(&mutex);
        }
    }
    /* Identify all the possible CUDA launch events without stream parameters,
     * they will not get involved with cuda graph */
    if (cbid == API_CUDA_cuLaunch  || cbid == API_CUDA_cuLaunchGrid) {
        /* cast params to cuLaunch_params since if we are here we know these are
         * the right parameters type */
        cuLaunch_params *p = (cuLaunch_params *)params;

        if (!is_exit) {
            pthread_mutex_lock(&mutex);

            try_to_instrument(p->f, ctx);
        } else {
            /* keep the old behavior (sync after each kernel is completed)
             * if cuda graph is not used. */
            /* if we are exiting a kernel launch:
             * 1. Wait until the kernel is completed using
             * cudaDeviceSynchronize()
             * 2. Get number of thread blocks in the kernel
             * 3. Print the thread instruction counters
             * 4. Release the lock*/
            CUDA_SAFECALL(cudaDeviceSynchronize());

            print_kernel_stats(p->f, ctx, 0);

            pthread_mutex_unlock(&mutex);
        }

    } else if (cbid == API_CUDA_cuGraphLaunch) {
            /* if we are exiting a cuda graph launch:
             * 1. Wait until the graph is completed using
             * cudaStreamSynchronize()
             * 2. Print the thread instruction counters
             * 3. Release the lock*/
            if (!is_exit) {
                pthread_mutex_lock(&mutex);
                return;
            }
            cuGraphLaunch_params *p = (cuGraphLaunch_params *)params;

            CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
            for (const auto& kernel: kernel_map) {
                print_kernel_stats(kernel.first, ctx, 0);
            }
            pthread_mutex_unlock(&mutex);
    } else if (cbid == API_CUDA_cuProfilerStart && is_exit) {
        if (!active_from_start) {
            active_region = true;
        }
    } else if (cbid == API_CUDA_cuProfilerStop && is_exit) {
        if (!active_from_start) {
            active_region = false;
        }
    }
}

void nvbit_at_term() {
    if (kernel_id >= MAX_NUM_KERNEL) {
        printf("We ran out of kernel_counters, please increase MAX_NUM_KERNEL to %d\n", kernel_id);
    }
    printf("Total app instructions: %ld\n", tot_app_instrs);
}
