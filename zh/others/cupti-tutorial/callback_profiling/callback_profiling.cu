//
// Copyright 2020-2025 NVIDIA Corporation. All rights reserved
//

// System headers
#include <list>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// CUPTI headers
#include <cupti.h>
#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>

// Helper headers
#include <helper_cupti.h>
#include <command_line_parser_util.h>
#include <cupti_range_profiler_util.h>

using namespace cupti::utils;

// Macros
#define METRIC_NAME "sm__ctas_launched.sum"

// Kernels
__global__ void VectorAdd(
    const int *pA,
    const int *pB,
    int *pC,
    int N
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] + pB[i];
    }
}

// Command line arguments struct
struct CommandLineArgs
{
    size_t device;
    size_t numRanges;
    std::vector<std::string> metrics;
    CUpti_ProfilerReplayMode replayMode;
    bool verbose;
};

// Profiling data struct
struct ProfilingData
{
    CUcontext context = nullptr;
    MetricEvaluator* metricEvaluator = nullptr;
    RangeProfiler* rangeProfiler = nullptr;

    size_t numRanges = 10;
    std::vector<std::string> metrics = {};

    std::vector<uint8_t> counterDataImage = {};
    std::list<MetricEvaluator::RangeInfo> rangeInfo = {};
};

// Parse command line arguments (forward declaration)
void ParseCommandLineArgs(
    int argc,
    char* argv[],
    CommandLineArgs& args
);

// Profiling callback handler
void ProfilingCallbackHandler(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    void *pCallbackData
)
{
    CUPTI_API_CALL(cuptiGetLastError());
    const CUpti_CallbackData *pCallbackInfo = (CUpti_CallbackData *)pCallbackData;
    ProfilingData* profilingData = (ProfilingData*)pUserData;
    switch (domain)
    {
        case CUPTI_CB_DOMAIN_DRIVER_API:
        {
            switch (callbackId)
            {
                case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
                {
                    if (pCallbackInfo->callbackSite == CUPTI_API_ENTER)
                    {
                        // Get context
                        CUcontext ctx = pCallbackInfo->context;

                        // Initialize metric evaluator
                        if (profilingData->metricEvaluator == nullptr) {
                            profilingData->metricEvaluator = new MetricEvaluator(ctx);
                        }

                        // Check if context is changed
                        if (profilingData->context && profilingData->context != ctx)
                        {
                            CUPTI_API_CALL(profilingData->rangeProfiler->DisableRangeProfiler());
                            delete profilingData->rangeProfiler;
                            profilingData->rangeProfiler = nullptr;
                        }

                        // Initialize range profiler
                        if (profilingData->rangeProfiler == nullptr)
                        {
                            std::unique_ptr<RangeProfiler> rangeProfiler = std::make_unique<RangeProfiler>(ctx, 0);
                            CUPTI_API_CALL(rangeProfiler->EnableRangeProfiler());
                            CUPTI_API_CALL(rangeProfiler->SetConfig(CUPTI_AutoRange, CUPTI_KernelReplay, profilingData->metrics, profilingData->counterDataImage, profilingData->numRanges));
                            profilingData->rangeProfiler = rangeProfiler.release();
                            profilingData->context = ctx;
                        }

                        // Start range profiler
                        CUPTI_API_CALL(profilingData->rangeProfiler->StartRangeProfiler());
                    }
                    else
                    {
                        // Stop range profiler
                        if (profilingData->rangeProfiler && profilingData->metricEvaluator)
                        {
                            CUPTI_API_CALL(profilingData->rangeProfiler->StopRangeProfiler());
                            CUPTI_API_CALL(profilingData->rangeProfiler->DecodeCounterData());

                            // Evaluate all ranges
                            std::vector<MetricEvaluator::RangeInfo> rangeInfos;
                            profilingData->metricEvaluator->evaluateAllRanges(profilingData->counterDataImage, profilingData->metrics, rangeInfos);
                            profilingData->rangeInfo.insert(profilingData->rangeInfo.end(), rangeInfos.begin(), rangeInfos.end());

                            // Initialize counter data image
                            CUPTI_API_CALL(profilingData->rangeProfiler->InitializeCounterDataImage(profilingData->counterDataImage));
                        }
                    }
                }
                break;
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
}

// Initialize vector
void InitializeVector(
    int *pVector,
    int N
)
{
    for (int i = 0; i < N; i++)
    {
        pVector[i] = i;
    }
}

// Clean up
static void CleanUp(
    int *pHostA,
    int *pHostB,
    int *pHostC,
    int *pDeviceA,
    int *pDeviceB,
    int *pDeviceC
)
{
    // Free host memory.
    if (pHostA)
    {
        free(pHostA);
    }
    if (pHostB)
    {
        free(pHostB);
    }
    if (pHostC)
    {
        free(pHostC);
    }

    // Free device memory.
    if (pDeviceA)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceA));
    }
    if (pDeviceB)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceB));
    }
    if (pDeviceC)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceC));
    }
}

// Do vector addition
void DoVectorAddition()
{
    int N = 50000;
    size_t size = N * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int* pHostA, * pHostB, * pHostC;
    int* pDeviceA, * pDeviceB, * pDeviceC;
    int i, sum;

    // Allocate input vectors pHostA and pHostB in host memory.
    pHostA = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostC);

    // Initialize input vectors
    InitializeVector(pHostA, N);
    InitializeVector(pHostB, N);
    memset(pHostC, 0, size);

    // Allocate vectors in device memory.
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, size));

    // Copy vectors from host memory to device memory.
    RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(pDeviceB, pHostB, size, cudaMemcpyHostToDevice));

    // Invoke kernel.
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel: blocks %d, thread/block %d\n", blocksPerGrid, threadsPerBlock);

    VectorAdd <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    RUNTIME_API_CALL(cudaGetLastError());

    // Copy result from device memory to host memory.
    // pHostC contains the result in host memory.
    RUNTIME_API_CALL(cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost));

    // Verify result.
    for (i = 0; i < N; ++i)
    {
        sum = pHostA[i] + pHostB[i];
        if (pHostC[i] != sum)
        {
            fprintf(stderr, "Error: result verification failed\n");
            exit(EXIT_FAILURE);
        }
    }

    CleanUp(pHostA, pHostB, pHostC, pDeviceA, pDeviceB, pDeviceC);
}

int main(
    int argc,
    char *argv[]
)
{
    int deviceCount, deviceNum = 0;
    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0)
    {
        printf("Warning: There is no device supporting CUDA.\nWaiving test.\n");
        exit(EXIT_WAIVED);
    }

    CUdevice cuDevice = 0;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));

    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);

    // Parse command line arguments
    CommandLineArgs args;
    ParseCommandLineArgs(argc, argv, args);

    // Check device support
    CUPTI_API_CALL(RangeProfiler::CheckDeviceSupport(args.device));

    // Initialize profiling data
    ProfilingData profilingData;
    profilingData.metrics = args.metrics;
    profilingData.numRanges = args.numRanges;

    // Subscribe to callbacks
    CUpti_SubscriberHandle subscriber;
    CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)ProfilingCallbackHandler, (void*)&profilingData));
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));

    // Launch workload
    for (int i = 0; i < 20; i++) {
        DoVectorAddition();
    }

    // Print profiling results
    std::vector<MetricEvaluator::RangeInfo> rangeInfos;
    std::transform(profilingData.rangeInfo.begin(), profilingData.rangeInfo.end(), std::back_inserter(rangeInfos), [](const MetricEvaluator::RangeInfo& rangeInfo) {
        return rangeInfo;
    });
    profilingData.metricEvaluator->printMetricData(rangeInfos);

    // Unsubscribe from callbacks
    CUPTI_API_CALL(cuptiUnsubscribe(subscriber));
}

std::vector<std::string> split(
    std::string str, char separator
)
{
    std::vector<std::string> result = {};
    std::string temp = "";

    for (int i = 0; i < (int)str.size(); i++)
    {
        if (str[i] != separator) {
            temp += str[i];
        } else {
            result.push_back(temp);
            temp = "";
        }
    }

    // Add the last token if there is one
    if (!temp.empty()) {
        result.push_back(temp);
    }
    return result;
}

void ParseCommandLineArgs(
    int argc,
    char* argv[],
    CommandLineArgs& args
)
{
    CommandLineParser parser;
    parser.addOption<size_t>("-d", "--device", "Device index", 0);
    parser.addOption<size_t>("-n", "--num-of-ranges", "Number of ranges to profile", 1);
    parser.addOption<std::string>("-m", "--metrics", "Metric name", "sm__ctas_launched.sum");
    parser.addOption<std::string>("-r", "--replay-mode", "Replay mode", "kernel");
    parser.addOption<bool>("-v", "--verbose", "Enable verbose mode", false);

    parser.parse(argc, argv);
    args.device = parser.get<size_t>("--device");
    args.numRanges = parser.get<size_t>("--num-of-ranges");
    std::string metricsStr = parser.get<std::string>("--metrics");
    args.metrics = split(metricsStr, ',');
    args.verbose = parser.get<bool>("--verbose");
    if (parser.get<std::string>("--replay-mode") == "kernel") {
        args.replayMode = CUPTI_KernelReplay;
    } else if (parser.get<std::string>("--replay-mode") == "user") {
        args.replayMode = CUPTI_UserReplay;
    }

    if (args.verbose)
    {
        std::cout << "Device: " << args.device << "\n";
        std::cout << "Metrics: ";
        for (const auto& metric : args.metrics) {
            std::cout << metric << " ";
        }
        std::cout << "\n";
        std::cout << "Replay mode: " << args.replayMode << "\n";
        std::cout << "Num of ranges: " << args.numRanges << "\n";
    }
}