/*
 *  Copyright 2024 NVIDIA Corporation. All rights reserved
 */

#include <atomic>
#include <chrono>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <thread>

#ifdef _WIN32
#define strdup _strdup
#endif

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

#include "range_profiling.h"

// Kernels
__global__
void vectorAdd(const int *pA, const int *pB, int *pC, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        pC[i] = pA[i] + pB[i];
    }
}

class VectorLaunchWorkLoad
{
    int m_numOfElements;
    int m_threadsPerBlock, m_blocksPerGrid;
    size_t m_size;

    int *pDeviceA, *pDeviceB, *pDeviceC;
    std::vector<int> pHostA, pHostB, pHostC;

public:
    VectorLaunchWorkLoad(int numElements = 50000, int threadsPerBlock = 256) :
        m_numOfElements(numElements), m_threadsPerBlock(threadsPerBlock)
    {
        m_size = m_numOfElements * sizeof(int);
        m_blocksPerGrid = (m_numOfElements + m_threadsPerBlock - 1) / m_threadsPerBlock;
        pHostA.resize(m_numOfElements);
        pHostB.resize(m_numOfElements);
        pHostC.resize(m_numOfElements);
    }

    ~VectorLaunchWorkLoad() {}

    void InitializeVector(std::vector<int>& pVector)
    {
        for (int i = 0; i < m_numOfElements; i++) {
            pVector[i] = i;
        }
    }

    void CleanUp()
    {
        // Free device memory.
        RUNTIME_API_CALL(cudaFree(pDeviceA));
        RUNTIME_API_CALL(cudaFree(pDeviceB));
        RUNTIME_API_CALL(cudaFree(pDeviceC));
    }

    void SetUp()
    {
        // Initialize input vectors
        InitializeVector(pHostA);
        InitializeVector(pHostB);
        std::fill(pHostC.begin(), pHostC.end(), 0);

        // Allocate vectors in device memory
        RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, m_size));
        RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, m_size));
        RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, m_size));

        // Copy vectors from host memory to device memory
        RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA.data(), m_size, cudaMemcpyHostToDevice));
        RUNTIME_API_CALL(cudaMemcpy(pDeviceB, pHostB.data(), m_size, cudaMemcpyHostToDevice));
    }

    void TearDown()
    {
        // Kernel Launch Verification
        // Copy result from device memory to host memory
        // pHostC contains the result in host memory
        RUNTIME_API_CALL(cudaMemcpy(pHostC.data(), pDeviceC, m_size, cudaMemcpyDeviceToHost));

        // Verify result
        int sum;
        for (int i = 0; i < m_numOfElements; ++i)
        {
            sum  = pHostA[i] + pHostB[i];
            if (pHostC[i] != sum)
            {
                fprintf(stderr, "Error: Result verification failed.\n");
                exit(EXIT_FAILURE);
            }
        }
        printf("Result verification passed.\n");
        CleanUp();
    }

    cudaError_t LaunchKernel()
    {
        vectorAdd<<<m_blocksPerGrid, m_threadsPerBlock>>>(pDeviceA, pDeviceB, pDeviceC, m_numOfElements);
        return cudaGetLastError();
    }
};

struct ParsedArgs
{
    int deviceIndex = 0;
    std::string rangeMode = "auto";
    std::string replayMode = "user";
    uint64_t maxRange = 20;
    std::vector<const char*> metrics =
    {
        "sm__ctas_launched.sum"
    };
};

ParsedArgs parseArgs(int argc, char *argv[]);

int main(int argc, char *argv[])
{
    ParsedArgs args = parseArgs(argc, argv);
    DRIVER_API_CALL(cuInit(0));

    printf("Starting Range Profiling\n");

    // Get the current ctx for the device
    CUdevice cuDevice;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, args.deviceIndex));

    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);

    if (computeCapabilityMajor < 7)
    {
        std::cerr << "Range Profiling is supported only on devices with compute capability 7.0 and above" << std::endl;
        exit(EXIT_FAILURE);
    }

    RangeProfilerConfig config;
    config.maxNumOfRanges = args.maxRange;
    config.minNestingLevel = 1;
    config.numOfNestingLevel = args.rangeMode == "user" ? 2 : 1;

    CuptiProfilerHostPtr pCuptiProfilerHost = std::make_shared<CuptiProfilerHost>();

    // Create a context
    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));
    RangeProfilerTargetPtr pRangeProfilerTarget = std::make_shared<RangeProfilerTarget>(cuContext, config);

    // Get chip name
    std::string chipName;
    CUPTI_API_CALL(RangeProfilerTarget::GetChipName(cuDevice, chipName));

    // Get Counter availability image
    std::vector<uint8_t> counterAvailabilityImage;
    CUPTI_API_CALL(RangeProfilerTarget::GetCounterAvailabilityImage(cuContext, counterAvailabilityImage));

    // Create config image
    std::vector<uint8_t> configImage;
    pCuptiProfilerHost->SetUp(chipName, counterAvailabilityImage);
    CUPTI_API_CALL(pCuptiProfilerHost->CreateConfigImage(args.metrics, configImage));

    // Set up the workload
    VectorLaunchWorkLoad vectorLaunchWorkLoad;
    vectorLaunchWorkLoad.SetUp();

    // Enable Range profiler
    CUPTI_API_CALL(pRangeProfilerTarget->EnableRangeProfiler());

    // Create CounterData Image
    std::vector<uint8_t> counterDataImage;
    CUPTI_API_CALL(pRangeProfilerTarget->CreateCounterDataImage(args.metrics, counterDataImage));

    // Set range profiler configuration
    printf("Range Mode: %s\n", args.rangeMode.c_str());
    printf("Replay Mode: %s\n", args.replayMode.c_str());
    CUPTI_API_CALL(pRangeProfilerTarget->SetConfig(
        args.rangeMode == "auto" ? CUPTI_AutoRange : CUPTI_UserRange,
        args.replayMode == "kernel" ? CUPTI_KernelReplay : CUPTI_UserReplay,
        configImage,
        counterDataImage
    ));

    do
    {
        // Start Range Profiling
        CUPTI_API_CALL(pRangeProfilerTarget->StartRangeProfiler());

        {
            // Push Range (Level 1)
            CUPTI_API_CALL(pRangeProfilerTarget->PushRange("VectorAdd"));

            // Launch CUDA workload
            RUNTIME_API_CALL(vectorLaunchWorkLoad.LaunchKernel());

            {
                // Push Range (Level 2)
                CUPTI_API_CALL(pRangeProfilerTarget->PushRange("Nested VectorAdd"));

                    RUNTIME_API_CALL(vectorLaunchWorkLoad.LaunchKernel());

                // Pop Range (Level 2)
                CUPTI_API_CALL(pRangeProfilerTarget->PopRange());
            }

            // Pop Range (Level 1)
            CUPTI_API_CALL(pRangeProfilerTarget->PopRange());
        }

        RUNTIME_API_CALL(vectorLaunchWorkLoad.LaunchKernel());

        // Stop Range Profiling
        CUPTI_API_CALL(pRangeProfilerTarget->StopRangeProfiler());
    }
    while (!pRangeProfilerTarget->IsAllPassSubmitted());

    // Get Profiler Data
    CUPTI_API_CALL(pRangeProfilerTarget->DecodeCounterData());

    // Evaluate the results
    size_t numRanges = 0;
    CUPTI_API_CALL(pCuptiProfilerHost->GetNumOfRanges(counterDataImage, numRanges));
    for (size_t rangeIndex = 0; rangeIndex < numRanges; ++rangeIndex)
    {
        CUPTI_API_CALL(pCuptiProfilerHost->EvaluateCounterData(rangeIndex, args.metrics, counterDataImage));
    }

    pCuptiProfilerHost->PrintProfilerRanges();

    // Clean up
    CUPTI_API_CALL(pRangeProfilerTarget->DisableRangeProfiler());
    pCuptiProfilerHost->TearDown();
    vectorLaunchWorkLoad.TearDown();

    DRIVER_API_CALL(cuCtxDestroy(cuContext));
    return 0;
}

void PrintHelp()
{
    printf("Usage:\n");
    printf("  Range Profiling:\n");
    printf("    ./range_profiling [args]\n");
    printf("        --device/-d <deviceIndex> : Device index to run the range profiling\n");
    printf("        --range/-r <auto/user> : Range profiling mode. auto: ranges are defined around each kernel user: user defined ranges (Push/Pop API)\n");
    printf("        --replay/-e <kernel/user> : Replay mode needed for multi-pass metrics. kernel: replay will be done by CUPTI internally user: replay done explicitly by user\n");
    printf("        --maxNumRanges/-n <maximum number of ranges stored in counterdata> : Maximum number of ranges stored in counterdata\n");
    printf("        --metrics/-m <metric1,metric2,...> : List of metrics to be collected\n");
}

ParsedArgs parseArgs(int argc, char *argv[])
{
    ParsedArgs args;
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--device" || arg == "-d")
        {
            args.deviceIndex = std::stoi(argv[++i]);
        }
        else if (arg == "--range" || arg == "-r")
        {
            args.rangeMode = argv[++i];
        }
        else if (arg == "--replay" || arg == "-e")
        {
            args.replayMode = argv[++i];
        }
        else if (arg == "--maxNumRanges" || arg == "-n")
        {
            args.maxRange = std::stoull(argv[++i]);
        }
        else if (arg == "--metrics" || arg == "-m")
        {
            std::stringstream ss(argv[++i]);
            std::string metric;
            args.metrics.clear();
            while (std::getline(ss, metric, ','))
            {
                args.metrics.push_back(strdup(metric.c_str()));
            }
        }
        else if (arg == "--help" || arg == "-h")
        {
            PrintHelp();
            exit(EXIT_SUCCESS);
        }
        else
        {
            fprintf(stderr, "Invalid argument: %s\n", arg.c_str());
            PrintHelp();
            exit(EXIT_FAILURE);
        }
    }
    return args;
}
