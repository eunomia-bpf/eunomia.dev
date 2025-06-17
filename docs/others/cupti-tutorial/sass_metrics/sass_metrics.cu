/*
 * Copyright 2023 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to demonstrate the usage of new sass metrics APIs.
 * This app will work on devices with compute capability 7.0 and higher.
 */

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>

// CUDA headers
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

// CUPTI headers
#include "helper_cupti.h"
#include <cupti_sass_metrics.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>

#define ARRAY_SIZE 32000
#define THREADS_PER_BLOCK 256

static std::map<uint64_t, std::string> metricIdToNameMap;

typedef enum
{
    VECTOR_ADD  = 0,
    VECTOR_SUB  = 1,
    VECTOR_MUL  = 2,
} VectorOperation;

/// Kernels
__global__ void
VectorAdd( const int *pA, const int *pB, int *pC, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] + pB[i];
    }
}

__global__ void
VectorSubtract( const int *pA, const int *pB, int *pC, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] - pB[i];
    }
}

__global__ void
VectorMultiply( const int *pA, const int *pB, int *pC, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] * pB[i];
    }
}

void ListSupportedMetrics(int deviceIndex);
void printSassData(CUpti_SassMetricsFlushData_Params* pParams);

static void
CleanUp(
    int *pHostA, int *pHostB, int *pHostC,
    int *pDeviceA, int *pDeviceB, int *pDeviceC)
{
    // Free host memory.
    if (pHostA) { free(pHostA); }
    if (pHostB) { free(pHostB); }
    if (pHostC) { free(pHostC); }

    // Free device memory.
    if (pDeviceA) { cudaFree(pDeviceA); }
    if (pDeviceB) { cudaFree(pDeviceB); }
    if (pDeviceC) { cudaFree(pDeviceC);}
}

static void
InitializeVector( int *pVector, int N)
{
    for (int i = 0; i < N; i++)
    {
        pVector[i] = i;
    }
}

static void
DoVectorOperation( const VectorOperation vectorOperation)
{
    int N = ARRAY_SIZE;
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = 0;
    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;
    size_t size = N * sizeof(int);
    int i, result = 0;

    CUcontext cuCtx;

    // Allocate input vectors pHostA and pHostB in host memory.
    pHostA = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostC);

    // Initialize input vectors.
    InitializeVector(pHostA, N);
    InitializeVector(pHostB, N);
    memset(pHostC, 0, size);

    // Allocate vectors in device memory.
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, size));

    cuCtxGetCurrent(&cuCtx);

    RUNTIME_API_CALL(cudaMemcpyAsync(pDeviceA, pHostA, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpyAsync(pDeviceB, pHostB, size, cudaMemcpyHostToDevice));

    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    if (vectorOperation == VECTOR_ADD)
    {
        printf("Launching VectorAdd\n");
        VectorAdd <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    }
    else if (vectorOperation == VECTOR_SUB)
    {
        printf("Launching VectorSubtract\n");
        VectorSubtract <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    }
    else if (vectorOperation == VECTOR_MUL)
    {
        printf("Launching VectorMultiply\n");
        VectorMultiply <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    }
    else
    {
        fprintf(stderr, "Error: invalid operation\n");
        exit(EXIT_FAILURE);
    }

    RUNTIME_API_CALL(cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost));

    // Verify result
    for (i = 0; i < N; ++i)
    {
        if (vectorOperation == VECTOR_ADD)
        {
            result = pHostA[i] + pHostB[i];
        }
        else if (vectorOperation == VECTOR_SUB)
        {
            result = pHostA[i] - pHostB[i];
        }
        else if (vectorOperation == VECTOR_MUL)
        {
            result = pHostA[i] * pHostB[i];
        }

        if (pHostC[i] != result)
        {
            fprintf(stderr, "Error: Result verification failed.\n");
            exit(EXIT_FAILURE);
        }
    }

    CleanUp(pHostA, pHostB, pHostC, pDeviceA, pDeviceB, pDeviceC);
}

void
CollectSassMetrics(uint32_t deviceNum, const CUcontext& cuCtx, const std::vector<std::string>& metrics, bool enableLazyPatching)
{
    CUpti_Device_GetChipName_Params getChipParams{ CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipParams.deviceIndex = deviceNum;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipParams));

    // 1) Create metric config list for configuring SASS metric data collection
    std::vector<CUpti_SassMetrics_Config> metricConfigs;
    {
        metricConfigs.resize(metrics.size());
        for (size_t i = 0; i < metrics.size(); ++i)
        {
            CUpti_SassMetrics_GetProperties_Params sassMetricsGetPropertiesParams { CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE };
            sassMetricsGetPropertiesParams.pChipName = getChipParams.pChipName;
            sassMetricsGetPropertiesParams.pMetricName = metrics[i].c_str();
            CUPTI_API_CALL(cuptiSassMetricsGetProperties(&sassMetricsGetPropertiesParams));
            metricConfigs[i].metricId = sassMetricsGetPropertiesParams.metric.metricId;
            metricConfigs[i].outputGranularity = CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_GPU;
            metricIdToNameMap.insert({ metricConfigs[i].metricId, metrics[i].c_str() });
            std::cout << "Metric Name: " << metrics[i] 
                      << ", MetricID: " << sassMetricsGetPropertiesParams.metric.metricId
                      << ", Metric Description: " << sassMetricsGetPropertiesParams.metric.pMetricDescription << "\n";
        }
    }

    // 2) Set SASS  configuration for the device
    CUpti_SassMetricsSetConfig_Params setConfigParams { CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE };
    setConfigParams.pConfigs = metricConfigs.data();
    setConfigParams.numOfMetricConfig = metricConfigs.size();
    setConfigParams.deviceIndex = deviceNum;
    CUPTI_API_CALL(cuptiSassMetricsSetConfig(&setConfigParams));

    // 3) Enable SASS Patching
    // if lazy  is disabled, all the functions in the modules will be patched in cuptiSassMetricsEnable() call.
    // if lazy  is enabled, the CUfunction/kernel which is called within cuptiSassMetricsEnable()/cuptiSassMetricsDisable() 
    // range will get patched at the very first execution.
    CUpti_SassMetricsEnable_Params sassMetricsEnableParams { CUpti_SassMetricsEnable_Params_STRUCT_SIZE };
    sassMetricsEnableParams.enableLazyPatching = enableLazyPatching;
    sassMetricsEnableParams.ctx = cuCtx;
    CUPTI_API_CALL(cuptiSassMetricsEnable(&sassMetricsEnableParams));
    printf("Enable SASS Patching\n");

    // VectorAdd will be patched here as the lazy  has been enabled.
    DoVectorOperation(VECTOR_ADD);

    {
        // 4) get number of metric instances and number of sass records collected.
        CUpti_SassMetricsGetDataProperties_Params sassMetricsGetDataPropertiesParams { CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE };
        sassMetricsGetDataPropertiesParams.ctx = cuCtx;
        CUPTI_API_CALL(cuptiSassMetricsGetDataProperties(&sassMetricsGetDataPropertiesParams));

        if (sassMetricsGetDataPropertiesParams.numOfInstances != 0 && sassMetricsGetDataPropertiesParams.numOfPatchedInstructionRecords != 0)
        {
            // 5) it is user responsibility to allocate memory for getting patched data. After call to cuptiSassGetMetricData() the records will be flushed.
            CUpti_SassMetricsFlushData_Params sassMetricsFlushDataParams { CUpti_SassMetricsFlushData_Params_STRUCT_SIZE };
            sassMetricsFlushDataParams.ctx = cuCtx;
            sassMetricsFlushDataParams.numOfInstances = sassMetricsGetDataPropertiesParams.numOfInstances;
            sassMetricsFlushDataParams.numOfPatchedInstructionRecords = sassMetricsGetDataPropertiesParams.numOfPatchedInstructionRecords;
            sassMetricsFlushDataParams.pMetricsData = (CUpti_SassMetrics_Data*)malloc(sassMetricsGetDataPropertiesParams.numOfPatchedInstructionRecords *
                                                                                                                sizeof(CUpti_SassMetrics_Data));
            for (size_t recordIndex = 0; recordIndex < sassMetricsGetDataPropertiesParams.numOfPatchedInstructionRecords; ++recordIndex)
            {
                sassMetricsFlushDataParams.pMetricsData[recordIndex].pInstanceValues =
                    (CUpti_SassMetrics_InstanceValue*)malloc(sassMetricsGetDataPropertiesParams.numOfInstances * sizeof(CUpti_SassMetrics_InstanceValue));
            }
            CUPTI_API_CALL(cuptiSassMetricsFlushData(&sassMetricsFlushDataParams));
            printSassData(&sassMetricsFlushDataParams);

            // Cleanup memory
            for (size_t recordIndex = 0; recordIndex < sassMetricsGetDataPropertiesParams.numOfPatchedInstructionRecords; ++recordIndex)
            {
                free((void*)sassMetricsFlushDataParams.pMetricsData[recordIndex].functionName);
                free(sassMetricsFlushDataParams.pMetricsData[recordIndex].pInstanceValues);
            }
            free(sassMetricsFlushDataParams.pMetricsData);
        }
    }

    // As this is the first vectorSubstract() kernel launch, the  will be done here.
    DoVectorOperation(VECTOR_SUB);

    // 6) Disable SASS Patching.
    // Note that with the call cuptiSassGetMetricData(), CUPTI reset all the records and here vector sub function is called after
    // cuptiSassGetMetricData() API, so CUPTI still have SASS records for Vector Sub function and as we are calling cuptiSassMetricsDisable()
    // API without calling cuptiSassGetMetricData() the data will be discarded here.
    CUpti_SassMetricsDisable_Params sassMetricsDisableParams {CUpti_SassMetricsDisable_Params_STRUCT_SIZE};
    sassMetricsDisableParams.ctx = cuCtx;
    CUPTI_API_CALL(cuptiSassMetricsDisable(&sassMetricsDisableParams));
    if (sassMetricsDisableParams.numOfDroppedRecords > 0)
    {
        // As Vector Sub kernel SASS metric data has not been flushed before calling cuptiSassMetricsDisable() API, they will be
        // marked as dropped records (number of patched instructions * number of instances for each record).
        std::cout << "Num of Dropped Records: " << sassMetricsDisableParams.numOfDroppedRecords << "\n";
    }
    printf("Disable SASS Patching\n");

    // Vector Mul function will not get patched as it is called outside enable/disable range.
    DoVectorOperation(VECTOR_MUL);

    // 7) Reset SASS  config for the device.
    CUpti_SassMetricsUnsetConfig_Params unsetConfigParams{ CUpti_SassMetricsUnsetConfig_Params_STRUCT_SIZE};
    unsetConfigParams.deviceIndex = deviceNum;
    CUPTI_API_CALL(cuptiSassMetricsUnsetConfig(&unsetConfigParams));
}

void 
Help(const char* sampleName)
{
    printf("For supported metrics list : %s [--deviceNum <deviceIndex>] --list\n", sampleName);
    printf("For SASS metrics collection : %s [--deviceNum <deviceIndex>] [--metric <metric names comma separated>] [--enableLazyPatching <enableLazyPatching(0/1)>]\n", sampleName);
}

int
main(int argc, char *argv[])
{
    std::vector<std::string> metrics;
    bool bEnableLazyPatching = true;
    char* pMetricName;
    int deviceNum = 0;
    bool bListMetrics = false;

    for (int i = 1; i < argc; ++i)
    {
        char *arg = argv[i];
        if (strcmp(arg, "--help") == 0)
        {
            Help(argv[0]);
            exit(EXIT_SUCCESS);
        }

        if (strcmp(arg, "--list") == 0)
        {
            bListMetrics = true;
            break;
        }

        if (strcmp(arg, "--deviceNum") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! Add device number for querying metrics details.\n");
                exit(EXIT_FAILURE);
            }
            deviceNum = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(arg, "--metric") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! Add metric names for querying metrics details.\n");
                exit(EXIT_FAILURE);
            }
            pMetricName = strtok(argv[i+1], ",");
            while (pMetricName != NULL)
            {
                metrics.push_back(pMetricName);
                pMetricName = strtok(NULL, ",");
            }
            i++;
        }
        else if (strcmp(arg, "--enableLazyPatching") == 0)
        {
            if (!argv[i + 1] || atoi(argv[i + 1]) < 0 || atoi(argv[i + 1]) > 1)
            {
                printf("ERROR!! Set 0/1 for disable/enable lazy patching mode.\n");
                exit(EXIT_FAILURE);
            }
            bEnableLazyPatching = atoi(argv[i + 1]);
            i++;
        }
        else
        {
            printf("Error!! Invalid Arguments\n");
            Help(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    printf("Device Num: %d\n", deviceNum);
    printf("Lazy Patching %s\n", (bEnableLazyPatching) ? "Enabled" : "Disabled");

    if (metrics.empty())
    {
        metrics.push_back("smsp__sass_inst_executed");
    }

    cudaDeviceProp prop;
    RUNTIME_API_CALL(cudaSetDevice(deviceNum));
    RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, deviceNum));
    printf("Device Name: %s\n", prop.name);
    printf("Device compute capability: %d.%d\n", prop.major, prop.minor);

    // Initialize profiler API and test device compatibility
    CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = deviceNum;
    params.api = CUPTI_PROFILER_SASS_METRICS;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Unable to profile on device " << deviceNum << ::std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
        }

        if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tWSL is not supported" << ::std::endl;
        }
        exit(EXIT_WAIVED);
    }


    CUcontext cuCtx;
    DRIVER_API_CALL(cuCtxCreate(&cuCtx, 0, deviceNum));

    if (bListMetrics){
        ListSupportedMetrics(deviceNum);
    } else{
        CollectSassMetrics(deviceNum, cuCtx, metrics, bEnableLazyPatching);
    }

    DRIVER_API_CALL(cuCtxDestroy(cuCtx));
    exit(EXIT_SUCCESS);
}

void ListSupportedMetrics(int deviceIndex)
{
    CUpti_Device_GetChipName_Params getChipParams{ CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipParams.deviceIndex = deviceIndex;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipParams));

    CUpti_SassMetrics_GetNumOfMetrics_Params getNumOfMetricParams = { CUpti_SassMetrics_GetNumOfMetrics_Params_STRUCT_SIZE };
    getNumOfMetricParams.pChipName = getChipParams.pChipName;
    CUPTI_API_CALL(cuptiSassMetricsGetNumOfMetrics(&getNumOfMetricParams));

    std::vector<CUpti_SassMetrics_MetricDetails> supportedMetrics(getNumOfMetricParams.numOfMetrics);
    CUpti_SassMetrics_GetMetrics_Params getMetricsParams = {CUpti_SassMetrics_GetMetrics_Params_STRUCT_SIZE};
    getMetricsParams.pChipName = getChipParams.pChipName;
    getMetricsParams.pMetricsList = supportedMetrics.data();
    getMetricsParams.numOfMetrics = supportedMetrics.size();
    CUPTI_API_CALL(cuptiSassMetricsGetMetrics(&getMetricsParams));
    for (size_t i = 0; i < supportedMetrics.size(); ++i)
    {
        std::cout << "Metric Name: " << supportedMetrics[i].pMetricName
                  << ", MetricID: " << supportedMetrics[i].metricId
                  << ", Metric Description: " << supportedMetrics[i].pMetricDescription << "\n";
    }
}

void printSassData(CUpti_SassMetricsFlushData_Params* pParams)
{
    using InstanceToMetricVal        = std::unordered_map<uint32_t, uint64_t>;                    // Key -> InstanceID
    using PcOffsetToInstanceTable    = std::map<uint32_t, InstanceToMetricVal>;                   // Key -> pcOffset
    using MetricToPcOffsetTable      = std::unordered_map<uint64_t, PcOffsetToInstanceTable>;     // Key -> metricId
    using FunctionToMetricTable      = std::unordered_map<std::string, MetricToPcOffsetTable>;    // Key -> function Name
    using ModuleToFunctionTable      = std::unordered_map<uint32_t, FunctionToMetricTable>;       // key -> module cubinCrc


    CUpti_SassMetrics_Data* pSassMetricData = pParams->pMetricsData;
    ModuleToFunctionTable moduleToFunctionTable;
    for (auto pcRecordIndex = 0; pcRecordIndex < pParams->numOfPatchedInstructionRecords; ++pcRecordIndex)
    {
        CUpti_SassMetrics_Data sassMetricData = pSassMetricData[pcRecordIndex];

        uint32_t cubinCrc = sassMetricData.cubinCrc;
        FunctionToMetricTable& functionToMetricTable = moduleToFunctionTable[cubinCrc];

        std::string functionName = sassMetricData.functionName;
        MetricToPcOffsetTable& metricToPcOffsetTable = functionToMetricTable[functionName];

        for (auto instanceIndex = 0; instanceIndex < pParams->numOfInstances; ++instanceIndex)
        {
            auto& metricValue = sassMetricData.pInstanceValues[instanceIndex];

            uint64_t metricId = metricValue.metricId;
            PcOffsetToInstanceTable& pcOffsetToInstanceTable = metricToPcOffsetTable[metricId];

            uint32_t pcOffset = sassMetricData.pcOffset;
            InstanceToMetricVal& instanceToMetricVal = pcOffsetToInstanceTable[pcOffset];

            instanceToMetricVal[instanceIndex] = metricValue.value;
        }
    }

    for (const auto& module : moduleToFunctionTable)
    {
        printf("\nModule cubinCrc: %u\n", module.first);
        for (const auto& function : module.second)
        {
            printf("Kernel Name: %s\n", function.first.c_str());
            for (const auto& metric : function.second)
            {
                printf("metric Name: %s\n", metricIdToNameMap[metric.first].c_str());
                for (const auto& pcOffset : metric.second)
                {
                    std::cout << "\t\t" << "[Inst] pcOffset: " << std::hex << "0x" << pcOffset.first;
                    std::cout << std::dec << "\tmetricValue: \t";

                    InstanceToMetricVal instanceToMetricVal = pcOffset.second;
                    for (const auto& instance : instanceToMetricVal)
                    {
                        std::cout << "[" << instance.first << "]: " << instance.second << "\t";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
        }
    }
}