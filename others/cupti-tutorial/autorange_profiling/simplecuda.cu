#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <cuda.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <Metric.h>
#include <Eval.h>
#include <FileOp.h>

#define NVPW_API_CALL(apiFuncCall)                                             \
do {                                                                           \
    NVPA_Status _status = apiFuncCall;                                         \
    if (_status != NVPA_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define CUPTI_API_CALL(apiFuncCall)                                            \
do {                                                                           \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

static int numRanges = 2;
#define METRIC_NAME "smsp__warps_launched.avg+"

// Device code
  __global__ void VecAdd(const int* A, const int* B, int* C, int N)
 {
     int i = blockDim.x * blockIdx.x + threadIdx.x;
     if (i < N)
         C[i] = A[i] + B[i];
 }

 // Device code
  __global__ void VecSub(const int* A, const int* B, int* C, int N)
 {
     int i = blockDim.x * blockIdx.x + threadIdx.x;
     if (i < N)
         C[i] = A[i] - B[i];
 }


static void initVec(int *vec, int n)
{
  for (int i=0; i< n; i++)
    vec[i] = i;
}

static void cleanUp(int *h_A, int *h_B, int *h_C, int *h_D, int *d_A, int *d_B, int *d_C, int *d_D)
{
  if (d_A)
    cudaFree(d_A);
  if (d_B)
    cudaFree(d_B);
  if (d_C)
    cudaFree(d_C);
  if (d_D)
    cudaFree(d_D);

  // Free host memory
  if (h_A)
    free(h_A);
  if (h_B)
    free(h_B);
  if (h_C)
    free(h_C);
  if (h_D)
    free(h_D);
}

static void VectorAddSubtract()
{
  int N = 50000;
  size_t size = N * sizeof(int);
  int threadsPerBlock = 0;
  int blocksPerGrid = 0;
  int *h_A, *h_B, *h_C, *h_D;
  int *d_A, *d_B, *d_C, *d_D;
  int i, sum, diff;

  // Allocate input vectors h_A and h_B in host memory
  h_A = (int*)malloc(size);
  h_B = (int*)malloc(size);
  h_C = (int*)malloc(size);
  h_D = (int*)malloc(size);

  // Initialize input vectors
  initVec(h_A, N);
  initVec(h_B, N);
  memset(h_C, 0, size);
  memset(h_D, 0, size);

  // Allocate vectors in device memory
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);
  cudaMalloc((void**)&d_D, size);

  // Copy vectors from host memory to device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Invoke kernel
  threadsPerBlock = 256;
  blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  printf("Launching kernel: blocks %d, thread/block %d\n",
         blocksPerGrid, threadsPerBlock);

  VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  VecSub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, N);

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost);

  // Verify result
  for (i = 0; i < N; ++i) {
    sum = h_A[i] + h_B[i];
    diff = h_A[i] - h_B[i];
    if (h_C[i] != sum || h_D[i] != diff) {
      fprintf(stderr, "error: result verification failed\n");
      exit(-1);
    }
  }

  cleanUp(h_A, h_B, h_C, h_D, d_A, d_B, d_C, d_D);
}

bool CreateCounterDataImage(
    std::vector<uint8_t>& counterDataImage,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImagePrefix)
{

    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    counterDataImageOptions.maxNumRanges = numRanges;
    counterDataImageOptions.maxNumRangeTreeNodes = numRanges;
    counterDataImageOptions.maxRangeNameLength = 64;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));

    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = {CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];

    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

    return true;
}

bool runTest(int deviceNum,
                       std::vector<uint8_t>& configImage,
                       std::vector<uint8_t>& counterDataScratchBuffer,
                       std::vector<uint8_t>& counterDataImage,
                       CUpti_ProfilerReplayMode profilerReplayMode,
                       CUpti_ProfilerRange profilerRange)
{

    CUdevice cuDevice;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));

    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));

    CUpti_Profiler_BeginSession_Params beginSessionParams = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};

    beginSessionParams.ctx = NULL;
    beginSessionParams.counterDataImageSize = counterDataImage.size();
    beginSessionParams.pCounterDataImage = &counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    beginSessionParams.range = profilerRange;
    beginSessionParams.replayMode = profilerReplayMode;
    beginSessionParams.maxRangesPerPass = numRanges;
    beginSessionParams.maxLaunchesPerPass = numRanges;

    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

    setConfigParams.pConfig = &configImage[0];
    setConfigParams.configSize = configImage.size();

    if(profilerReplayMode == CUPTI_KernelReplay)    /* Profile in KernelReplayMode */
    {
        setConfigParams.passIndex = 0;
        CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
        CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
        VectorAddSubtract();
        CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
    }
    else if(profilerReplayMode == CUPTI_UserReplay) /* Profiler in UserReplayMode */
    {
        setConfigParams.passIndex = 0;
        CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
        /* User takes the resposiblity of replaying the kernel launches */
        CUpti_Profiler_BeginPass_Params beginPassParams = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
        CUpti_Profiler_EndPass_Params endPassParams = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE};
        do
        {
            CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
            {
                CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
                VectorAddSubtract();
                CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
            }
            CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
        }while(!endPassParams.allPassesSubmitted);
        CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
    }
    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

    DRIVER_API_CALL(cuCtxDestroy(cuContext));

    return true;
}

int main(int argc, char* argv[])
{
    std::vector<std::string> metricNames;
    std::vector<uint8_t> counterDataImagePrefix;
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataImage;
    std::vector<uint8_t> counterDataScratchBuffer;
    std::string CounterDataFileName("SimpleCupti.counterdata");
    std::string CounterDataSBFileName("SimpleCupti.counterdataSB");
    CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_KernelReplay;
    CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;
    int deviceCount, deviceNum;
    char* metricName;

    printf("Usage: %s [device_num] [metric_names comma separated]\n", argv[0]);

    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        return -2;
    }

    if (argc > 1)
        deviceNum = atoi(argv[1]);
    else
        deviceNum = 0;
    printf("CUDA Device Number: %d\n", deviceNum);

    // Get the names of the metrics to collect
    if (argc > 2) {
        metricName = strtok(argv[2], ",");
        while(metricName != NULL)
        {
            metricNames.push_back(metricName);
            metricName = strtok(NULL, ",");
        }
    }
    else {
        metricNames.push_back(METRIC_NAME);
    }

    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
    /* Get chip name for the cuda  device */
    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = deviceNum;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    std::string chipName(getChipNameParams.pChipName);

    /* Generate configuration for metrics, this can also be done offline*/
    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

    if (metricNames.size()) {
        if(!NV::Metric::Config::GetConfigImage(chipName, metricNames, configImage))
        {
            std::cout << "Failed to create configImage" << std::endl;
            exit(-1);
        }
        if(!NV::Metric::Config::GetCounterDataPrefixImage(chipName, metricNames, counterDataImagePrefix))
        {
            std::cout << "Failed to create counterDataImagePrefix" << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cout << "No metrics provided to profile" << std::endl;
        exit(-1);
    }

    if(!CreateCounterDataImage(counterDataImage, counterDataScratchBuffer, counterDataImagePrefix))
    {
        std::cout << "Failed to create counterDataImage" << std::endl;
        exit(-1);
    }

    if(!runTest(deviceNum, configImage, counterDataScratchBuffer, counterDataImage, profilerReplayMode, profilerRange))
    {
        std::cout << "Failed to run sample" << std::endl;
        exit(-1);
    }
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));

    /* Dump counterDataImage in file */
    WriteBinaryFile(CounterDataFileName.c_str(), counterDataImage);
    WriteBinaryFile(CounterDataSBFileName.c_str(), counterDataScratchBuffer);

    /* Evaluation of metrics collected in counterDataImage, this can also be done offline*/
    NV::Metric::Eval::PrintMetricValues(chipName, counterDataImage, metricNames);

    return 0;
}