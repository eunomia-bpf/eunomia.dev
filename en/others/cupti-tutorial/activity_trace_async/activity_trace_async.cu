/*
 * Copyright 2011-2022 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print a trace of CUDA API and GPU activity
 */

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// CUPTI headers
#include "helper_cupti_activity.h"

// Macros
#define COMPUTE_N 50000

// Kernels
__global__ void
VectorAdd(
    const int* pA,
    const int* pB,
    int *pC,
    int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] + pB[i];
    }
}

__global__ void
VectorSubtract(
    const int* pA,
    const int* pB,
    int *pC,
    int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] - pB[i];
    }
}

// Functions
static void
DoPass(
    cudaStream_t stream)
{
    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;
    size_t size = COMPUTE_N * sizeof(int);
    int threadsPerBlock = 256;
    int blocksPerGrid = 0;

    // Allocate input vectors pHostA and pHostB in host memory.
    // Don't bother to initialize.
    pHostA = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);


    // Allocate vectors in device memory.
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, size));

    RUNTIME_API_CALL(cudaMemcpyAsync(pDeviceA, pHostA, size, cudaMemcpyHostToDevice, stream));
    RUNTIME_API_CALL(cudaMemcpyAsync(pDeviceB, pHostB, size, cudaMemcpyHostToDevice, stream));

    blocksPerGrid = (COMPUTE_N + threadsPerBlock - 1) / threadsPerBlock;

    VectorAdd <<< blocksPerGrid, threadsPerBlock, 0, stream >>> (pDeviceA, pDeviceB, pDeviceC, COMPUTE_N);
    RUNTIME_API_CALL(cudaGetLastError());

    VectorSubtract <<< blocksPerGrid, threadsPerBlock, 0, stream >>> (pDeviceA, pDeviceB, pDeviceC, COMPUTE_N);
    RUNTIME_API_CALL(cudaGetLastError());

    RUNTIME_API_CALL(cudaMemcpyAsync(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost, stream));

    if (stream == 0)
    {
        RUNTIME_API_CALL(cudaDeviceSynchronize());
    }
    else
    {
        RUNTIME_API_CALL(cudaStreamSynchronize(stream));
    }

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

void CUPTIAPI CallbackHandler(CUpti_CallbackDomain domain, CUpti_CallbackId callbackId)
{
    CUPTI_API_CALL(cuptiGetLastError());

    switch (domain)
    {
        case CUPTI_CB_DOMAIN_RESOURCE:
            switch (callbackId)
            {
                CUptiResult result;
                case CUPTI_CBID_RESOURCE_CU_INIT_FINISHED:
                    result = cuptiActivityEnableHWTrace(1);
                    if (result != CUPTI_SUCCESS)
                    {
                        const char *errstr;
                        cuptiGetResultString(result, &errstr);
                        printf("cuptiActivityEnableHWTrace failed with error %s. Legacy tracing will be continued.\n", errstr);
                    }
                    break;
                default:
                    break;
            }
            break;
        default:
            break;
    }
    return;
}

static void
SetupCupti()
{
    UserData *pUserData = (UserData *)malloc(sizeof(UserData));
    MEMORY_ALLOCATION_CALL(pUserData);

    memset(pUserData, 0, sizeof(UserData));
    pUserData->pPostProcessActivityRecords = NULL;
    pUserData->printActivityRecords        = 1;

    // Common CUPTI Initialization
    InitCuptiTrace(pUserData, (void *)CallbackHandler, stdout);

    // Device activity record is created when CUDA initializes, so we
    // want to enable it before cuInit() or any CUDA runtime call.
    CUPTI_API_CALL_VERBOSE(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
    // Enable all other activity record kinds.
    CUPTI_API_CALL_VERBOSE(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
    CUPTI_API_CALL_VERBOSE(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_API_CALL_VERBOSE(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_API_CALL_VERBOSE(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_API_CALL_VERBOSE(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_API_CALL_VERBOSE(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
    CUPTI_API_CALL_VERBOSE(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
    CUPTI_API_CALL_VERBOSE(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_API_CALL_VERBOSE(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
}

void printUsage() {
    std::cout << "Usage: \n"
              << "  ./activity_trace_async -e [ --enableHwTrace ] <value>\n"
              << "  Non-zero value to enable, 0 to disable (disabled by default)\n";
}

int
main(
    int argc,
    char *argv[])
{
    CUdevice device;
    char deviceName[256];
    int deviceId = 0, deviceCount = 0;
    int enableHwTrace = 0;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--enableHwTrace" || arg == "-e")
        {
            if (i + 1 >= argc)
            {
                std::cout << "Missing value for " << arg << " option." << std::endl;
                printUsage();
                return 1;
            }

            enableHwTrace = std::atoi(argv[++i]);
        }
        else
        {
            std::cout << "Unknown option: " << arg << std::endl;
            printUsage();
            return 1;
        }
    }

    SetupCupti();

    // Enable CUPTI callback on CUDA Initialization Finished if hardware trace is enabled
    if (enableHwTrace)
    {
        CUPTI_API_CALL_VERBOSE(cuptiEnableCallback(1, globals.subscriberHandle, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CU_INIT_FINISHED));
    }

    // Intialize CUDA
    DRIVER_API_CALL(cuInit(0));

    RUNTIME_API_CALL(cudaGetDeviceCount(&deviceCount));

    for (deviceId = 0; deviceId < deviceCount; deviceId++)
    {
        DRIVER_API_CALL(cuDeviceGet(&device, deviceId));
        DRIVER_API_CALL(cuDeviceGetName(deviceName, 256, device));
        printf("Device Name: %s\n", deviceName);

        RUNTIME_API_CALL(cudaSetDevice(deviceId));

        // DoPass with default stream
        DoPass(0);

        // DoPass with user stream
        cudaStream_t stream;
        RUNTIME_API_CALL(cudaStreamCreate(&stream));
        DoPass(stream);

        RUNTIME_API_CALL(cudaDeviceSynchronize());

        RUNTIME_API_CALL(cudaDeviceReset());
    }

    DeInitCuptiTrace();

    if (enableHwTrace)
    {
        cuptiActivityEnableHWTrace(0);
    }

    exit(EXIT_SUCCESS);
}
