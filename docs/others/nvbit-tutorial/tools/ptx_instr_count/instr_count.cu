// POC: NVBit tool with hand-written PTX device functions
// Host-side code (based on instr_count_bb, simplified)
// The device function "count_instrs" comes from inject_funcs.ptx (NOT .cu)

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

uint32_t kernel_id = 0;
uint64_t tot_app_instrs = 0;

/* kernel instruction counter, updated by the GPU threads */
__managed__ uint64_t counter = 0;

/* global control variables */
uint32_t start_grid_num = 0;
uint32_t end_grid_num = UINT32_MAX;
int verbose = 0;
int count_warp_level = 1;
int active_from_start = 1;
bool mangled = false;
bool active_region = true;

pthread_mutex_t mutex;

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    GET_VAR_INT(start_grid_num, "START_GRID_NUM", 0,
                "Beginning of the kernel gird launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(end_grid_num, "END_GRID_NUM", UINT32_MAX,
                "End of the kernel grid launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 1,
                "Count warp level or thread level instructions");
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
    printf("  [PTX POC] Device functions from hand-written PTX!\n");
    printf("%s\n", pad.c_str());
}

std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);
    related_functions.push_back(func);

    for (auto f : related_functions) {
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        const CFG_t &cfg = nvbit_get_CFG(ctx, f);
        if (cfg.is_degenerate) {
            printf(
                "Warning: Function %s is degenerated, we can't compute basic "
                "blocks statically",
                nvbit_get_func_name(ctx, f));
        }

        if (verbose) {
            printf("inspecting %s - number basic blocks %ld\n",
                   nvbit_get_func_name(ctx, f), cfg.bbs.size());
        }

        /* Iterate on basic block and inject the first instruction */
        for (auto &bb : cfg.bbs) {
            Instr *i = bb->instrs[0];
            /* inject device function — "count_instrs" is defined in
             * inject_funcs.ptx (hand-written PTX, NOT CUDA C!) */
            nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
            nvbit_add_call_arg_const_val32(i, bb->instrs.size());
            nvbit_add_call_arg_const_val32(i, count_warp_level);
            nvbit_add_call_arg_const_val64(i, (uint64_t)&counter);
        }
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {

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
            pthread_mutex_lock(&mutex);
            instrument_function_if_needed(ctx, func);

            if (active_from_start) {
                if (kernel_id >= start_grid_num && kernel_id < end_grid_num) {
                    active_region = true;
                } else {
                    active_region = false;
                }
            }

            if (active_region) {
                nvbit_enable_instrumented(ctx, func, true);
            } else {
                nvbit_enable_instrumented(ctx, func, false);
            }

            counter = 0;
        } else {
            CUDA_SAFECALL(cudaDeviceSynchronize());
            tot_app_instrs += counter;
            int num_ctas = 0;
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
            printf(
                "kernel %d - %s - #thread-blocks %d,  kernel "
                "instructions %ld, total instructions %ld\n",
                kernel_id++, nvbit_get_func_name(ctx, func, mangled), num_ctas,
                counter, tot_app_instrs);
            pthread_mutex_unlock(&mutex);
        }
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
    printf("Total app instructions: %ld\n", tot_app_instrs);
}
