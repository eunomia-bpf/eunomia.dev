// Copyright 2021-2022 NVIDIA Corporation. All rights reserved
//
// This sample demostrates using the profiler API in injection mode.
// Build this file as a shared object, and set environment variable
// CUDA_INJECTION64_PATH to the full path to the .so.
//
// CUDA will load the object during initialization and will run
// the function called 'InitializeInjection'.
//
// After the initialization routine  returns, the application resumes running,
// with the registered callbacks triggering as expected.  These callbacks
// are used to start a Profiler API session using Kernel Replay and
// Auto Range modes.
//
// A configurable number of kernel launches (default 10) are run
// under one session.  Before the 11th kernel launch, the callback
// ends the session, prints metrics, and starts a new session.
//
// An atexit callback is also used to ensure that any partial sessions
// are handled when the target application exits.
//
// This code supports multiple contexts and multithreading through
// locking shared data structures.

#include <list>
#include <mutex>
#include <unordered_map>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime_api.h>

// CUPTI headers
#include <cupti.h>

#include "helper_cupti.h"
#include "cupti_range_profiler_util.h"
#include "cupti_profiler_host_util.h"

using namespace cupti::utils;

// Macros
// Export InitializeInjection symbol.
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define HIDDEN
#else
#define DLLEXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))
#endif

// Profiler API data, per-context.
struct CtxProfilerData
{
    CUcontext       ctx = nullptr;
    int             deviceId = 0;
    char            deviceName[DEV_NAME_LEN];
    int             maxNumRanges = 10;
    bool            isActive = false;
    std::vector<uint8_t> counterDataImage = {};

    std::unique_ptr<cupti::utils::MetricEvaluator> metricEvaluator = nullptr;
    std::unique_ptr<cupti::utils::RangeProfiler> rangeProfiler = nullptr;
    std::list<cupti::utils::MetricEvaluator::RangeInfo> rangeInfo = {};
};

// Track per-context profiler API data in a shared map.
std::mutex ctxDataMutex;
std::unordered_map<CUcontext, std::unique_ptr<CtxProfilerData>> contextData;
CtxProfilerData* activeCtxProfilerData = nullptr;

// List of metrics to collect.
std::vector<std::string> metricNames;

// Print session data
static void PrintData(
    CtxProfilerData &ctxProfilerData
)
{
    DRIVER_API_CALL(cuDeviceGetName(ctxProfilerData.deviceName, DEV_NAME_LEN, 0));
    std::cout << std::endl
              << "Context " << ctxProfilerData.ctx
              << ", device " << ctxProfilerData.deviceId
              << " (" << ctxProfilerData.deviceName << ")"
              << ":" << std::endl;

    std::vector<MetricEvaluator::RangeInfo> rangeInfos(ctxProfilerData.rangeInfo.size());
    std::transform(ctxProfilerData.rangeInfo.begin(), ctxProfilerData.rangeInfo.end(), rangeInfos.begin(), [](MetricEvaluator::RangeInfo& rangeInfo) {
        return rangeInfo;
    });
    ctxProfilerData.metricEvaluator->printMetricData(rangeInfos);
}

// Callback handler
void ProfilerCallbackHandler(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    void const *pCallbackData
)
{
    CUptiResult res = CUPTI_SUCCESS;
    CUPTI_API_CALL(cuptiGetLastError());
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API)
    {
        // For a driver call to launch a kernel:
        if (callbackId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)
        {
            CUpti_CallbackData const *pData = static_cast<CUpti_CallbackData const *>(pCallbackData);
            CUcontext ctx = pData->context;

            // On entry
            if (pData->callbackSite == CUPTI_API_ENTER)
            {
                std::lock_guard<std::mutex> lock(ctxDataMutex);
                if (contextData.count(ctx) > 0)
                {
                    // Check if context is changed
                    if (activeCtxProfilerData != nullptr && activeCtxProfilerData->ctx && activeCtxProfilerData->ctx != ctx)
                    {
                        activeCtxProfilerData->rangeProfiler->StopRangeProfiler();
                        activeCtxProfilerData->rangeProfiler->DecodeCounterData();
                        std::vector<MetricEvaluator::RangeInfo> rangeInfos;
                        activeCtxProfilerData->metricEvaluator->evaluateAllRanges(activeCtxProfilerData->counterDataImage, metricNames, rangeInfos);

                        // Store range info
                        std::for_each(rangeInfos.begin(), rangeInfos.end(), [](MetricEvaluator::RangeInfo& rangeInfo) {
                            activeCtxProfilerData->rangeInfo.push_back(rangeInfo);
                        });

                        // Disable range profiler
                        CUPTI_API_CALL(activeCtxProfilerData->rangeProfiler->DisableRangeProfiler());
                        activeCtxProfilerData->rangeProfiler.reset();
                        activeCtxProfilerData->isActive = false;
                        activeCtxProfilerData = nullptr;

                        // Set active context
                        CtxProfilerData* ctxProfilerData = contextData[ctx].get();
                        activeCtxProfilerData = ctxProfilerData;
                    }

                    // Initialize range profiler
                    if (activeCtxProfilerData->rangeProfiler == nullptr)
                    {
                        std::unique_ptr<RangeProfiler> rangeProfiler = std::make_unique<RangeProfiler>(ctx, 0);
                        CUPTI_API_CALL(rangeProfiler->EnableRangeProfiler());
                        CUPTI_API_CALL(rangeProfiler->SetConfig(
                            CUPTI_AutoRange,
                            CUPTI_KernelReplay,
                            metricNames,
                            activeCtxProfilerData->counterDataImage,
                            activeCtxProfilerData->maxNumRanges
                        ));
                        CUPTI_API_CALL(rangeProfiler->StartRangeProfiler());
                        activeCtxProfilerData->rangeProfiler = std::move(rangeProfiler);
                        activeCtxProfilerData->isActive = true;
                    }

                    // Decode collected counter data
                    CUPTI_API_CALL(activeCtxProfilerData->rangeProfiler->DecodeCounterData());
                    std::vector<MetricEvaluator::RangeInfo> rangeInfos;
                    activeCtxProfilerData->metricEvaluator->evaluateAllRanges(activeCtxProfilerData->counterDataImage, metricNames, rangeInfos);

                    std::for_each(rangeInfos.begin(), rangeInfos.end(), [](MetricEvaluator::RangeInfo& rangeInfo) {
                        activeCtxProfilerData->rangeInfo.push_back(rangeInfo);
                    });

                    // Reset counter data image
                    CUPTI_API_CALL(activeCtxProfilerData->rangeProfiler->InitializeCounterDataImage(activeCtxProfilerData->counterDataImage));
                }
            }
        }
    }
    else if (domain == CUPTI_CB_DOMAIN_RESOURCE)
    {
        // When a context is created, check to see whether the device is compatible with the Profiler API
        if (callbackId == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)
        {
            CUpti_ResourceData const *pResourceData = static_cast<CUpti_ResourceData const *>(pCallbackData);
            CUcontext ctx = pResourceData->context;

            std::lock_guard<std::mutex> lock(ctxDataMutex);

            // Disable range profiler for previous context
            if (activeCtxProfilerData != nullptr && activeCtxProfilerData->isActive)
            {
                CUPTI_API_CALL(activeCtxProfilerData->rangeProfiler->StopRangeProfiler());
                CUPTI_API_CALL(activeCtxProfilerData->rangeProfiler->DecodeCounterData());
                std::vector<MetricEvaluator::RangeInfo> rangeInfos;
                activeCtxProfilerData->metricEvaluator->evaluateAllRanges(activeCtxProfilerData->counterDataImage, metricNames, rangeInfos);

                std::for_each(rangeInfos.begin(), rangeInfos.end(), [](MetricEvaluator::RangeInfo& rangeInfo) {
                    activeCtxProfilerData->rangeInfo.push_back(rangeInfo);
                });

                CUPTI_API_CALL(activeCtxProfilerData->rangeProfiler->InitializeCounterDataImage(activeCtxProfilerData->counterDataImage));
                CUPTI_API_CALL(activeCtxProfilerData->rangeProfiler->DisableRangeProfiler());
                activeCtxProfilerData->rangeProfiler.reset();
                activeCtxProfilerData->isActive = false;
                activeCtxProfilerData = nullptr;
            }

            // Configure handler for new context under lock
            std::unique_ptr<CtxProfilerData> ctxProfilerData = std::make_unique<CtxProfilerData>();
            ctxProfilerData->ctx = ctx;
            RUNTIME_API_CALL(cudaGetDevice(&(ctxProfilerData->deviceId)));
            DRIVER_API_CALL(cuDeviceGetName(ctxProfilerData->deviceName, DEV_NAME_LEN, 0));

            // Check if device is supported for profiling
            CUPTI_API_CALL(RangeProfiler::CheckDeviceSupport(ctxProfilerData->deviceId));

            // Initialize metric evaluator
            if (ctxProfilerData->metricEvaluator == nullptr) {
                ctxProfilerData->metricEvaluator = std::make_unique<MetricEvaluator>(ctx);
            }

            // Initialize range profiler
            if (ctxProfilerData->rangeProfiler == nullptr)
            {
                std::unique_ptr<RangeProfiler> rangeProfiler = std::make_unique<RangeProfiler>(ctx, 0);
                CUPTI_API_CALL(rangeProfiler->EnableRangeProfiler());
                CUPTI_API_CALL(rangeProfiler->SetConfig(
                    CUPTI_AutoRange,
                    CUPTI_KernelReplay,
                    metricNames,
                    ctxProfilerData->counterDataImage,
                    ctxProfilerData->maxNumRanges
                ));
                CUPTI_API_CALL(rangeProfiler->StartRangeProfiler());
                ctxProfilerData->rangeProfiler = std::move(rangeProfiler);
                ctxProfilerData->isActive = true;
                contextData[ctx] = std::move(ctxProfilerData);
                activeCtxProfilerData = contextData[ctx].get();
            }
        }
        else if (callbackId == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING)
        {
            CUpti_ResourceData const *pResourceData = static_cast<CUpti_ResourceData const *>(pCallbackData);
            CUcontext ctx = pResourceData->context;

            std::lock_guard<std::mutex> lock(ctxDataMutex);

            CtxProfilerData* ctxProfilerData = contextData[ctx].get();
            if (ctxProfilerData->isActive)
            {
                CUPTI_API_CALL(ctxProfilerData->rangeProfiler->StopRangeProfiler());
                CUPTI_API_CALL(ctxProfilerData->rangeProfiler->DecodeCounterData());
                std::vector<MetricEvaluator::RangeInfo> rangeInfos;
                ctxProfilerData->metricEvaluator->evaluateAllRanges(ctxProfilerData->counterDataImage, metricNames, rangeInfos);
                std::for_each(rangeInfos.begin(), rangeInfos.end(), [&ctxProfilerData](MetricEvaluator::RangeInfo& rangeInfo) {
                    ctxProfilerData->rangeInfo.push_back(rangeInfo);
                });

                CUPTI_API_CALL(ctxProfilerData->rangeProfiler->InitializeCounterDataImage(ctxProfilerData->counterDataImage));
                CUPTI_API_CALL(ctxProfilerData->rangeProfiler->DisableRangeProfiler());
                ctxProfilerData->rangeProfiler.reset();
                ctxProfilerData->isActive = false;
            }
        }
    }

    return;
}

void EndExecution()
{
    std::cout << "Ending execution" << std::endl;
    for (auto& ctxProfilerDataPair : contextData)
    {
        CtxProfilerData* ctxProfilerData = ctxProfilerDataPair.second.get();
        if (ctxProfilerData && ctxProfilerData->isActive)
        {
            CUPTI_API_CALL(ctxProfilerData->rangeProfiler->StopRangeProfiler());
            CUPTI_API_CALL(ctxProfilerData->rangeProfiler->DecodeCounterData());
            std::vector<MetricEvaluator::RangeInfo> rangeInfos;
            ctxProfilerData->metricEvaluator->evaluateAllRanges(ctxProfilerData->counterDataImage, metricNames, rangeInfos);
            std::for_each(rangeInfos.begin(), rangeInfos.end(), [&ctxProfilerData](MetricEvaluator::RangeInfo& rangeInfo) {
                ctxProfilerData->rangeInfo.push_back(rangeInfo);
            });
        }

        PrintData(*ctxProfilerData);
    }
}

// Register callbacks for several points in target application execution
void RegisterCallbacks()
{
    // One subscriber is used to register multiple callback domains
    CUpti_SubscriberHandle subscriber;
    CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)ProfilerCallbackHandler, NULL));
    // Runtime callback domain is needed for kernel launch callbacks
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    // Resource callback domain is needed for context creation callbacks
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING));

    // Register callback for application exit
    atexit(EndExecution);
}

static bool injectionInitialized = false;

// InitializeInjection will be called by the driver when the tool is loaded
// by CUDA_INJECTION64_PATH
extern "C" DLLEXPORT
int InitializeInjection()
{
    if (injectionInitialized == false)
    {
        injectionInitialized = true;

        // Read in optional list of metrics to gather
        char *pMetricEnv = getenv("INJECTION_METRICS");
        if (pMetricEnv != NULL)
        {
            char * tok = strtok(pMetricEnv, " ;,");
            do
            {
                std::cout << "Requesting metric '" << tok << "'" << std::endl;
                metricNames.push_back(std::string(tok));
                tok = strtok(NULL, " ;,");
            } while (tok != NULL);
        }
        else
        {
            metricNames.push_back("sm__cycles_elapsed.avg");
            metricNames.push_back("smsp__sass_thread_inst_executed_op_dadd_pred_on.avg");
            metricNames.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.avg");
        }

        // Subscribe to some callbacks
        RegisterCallbacks();
    }
    return 1;
}
