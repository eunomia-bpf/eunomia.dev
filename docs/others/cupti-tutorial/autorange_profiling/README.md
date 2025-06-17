# CUPTI Autorange Profiling Tutorial

## Introduction

Profiling CUDA applications typically requires manually instrumenting your code to define which regions to profile. However, CUPTI's autorange profiling feature simplifies this process by automatically detecting and profiling kernel launches. This tutorial demonstrates how to use this powerful feature to collect performance metrics without modifying your kernels.

## What You'll Learn

- How to set up automatic profiling of CUDA kernels
- Collecting performance metrics without manual instrumentation
- Using NVIDIA's advanced profiling APIs
- Interpreting the collected metrics for performance analysis

## Understanding Autorange Profiling

Autorange profiling automatically detects when CUDA kernels are launched and collects performance metrics for each kernel. This is particularly useful when:

- You're analyzing third-party code where you can't add instrumentation
- You want to profile all kernels in an application without manual intervention
- You need a quick performance overview without modifying source code

## Code Walkthrough

### 1. Setting Up the Profiling Environment

First, we need to initialize the CUPTI profiler and configure it for autorange profiling:

```cpp
// Initialize CUPTI and NVPW libraries
NVPW_InitializeHost_Params initializeHostParams = {NVPW_InitializeHost_Params_STRUCT_SIZE};
NVPW_InitializeHost(&initializeHostParams);

// Create counter data image
NV_CUPTI_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

// Set up configuration for the metrics we want to collect
const char* metricName = METRIC_NAME; // Default is "smsp__warps_launched.avg+"
```

### 2. Creating the Counter Data Image

The counter data image is where CUPTI stores the raw performance data:

```cpp
// Create counter data image that will store the collected metrics
CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
counterDataImageOptions.size = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
counterDataImageOptions.pCounterDataPrefix = NULL;
counterDataImageOptions.counterDataPrefixSize = 0;
counterDataImageOptions.maxNumRanges = 2;
counterDataImageOptions.maxNumRangeTreeNodes = 2;
counterDataImageOptions.maxRangeNameLength = 64;

CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
    CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};
calculateSizeParams.pOptions = &counterDataImageOptions;
NV_CUPTI_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

// Allocate memory for the counter data image
counterDataImage = (uint8_t*)malloc(calculateSizeParams.counterDataImageSize);
```

### 3. Configuring the Metrics to Collect

We need to specify which metrics we want to collect:

```cpp
// Create configuration for the metrics
CUpti_Profiler_BeginSession_Params beginSessionParams = {
    CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
beginSessionParams.ctx = NULL;
beginSessionParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
beginSessionParams.pCounterDataImage = counterDataImage;
beginSessionParams.counterDataScratchBufferSize = calculateSizeParams.counterDataScratchBufferSize;
beginSessionParams.pCounterDataScratchBuffer = counterDataScratchBuffer;
beginSessionParams.range = CUPTI_AutoRange;
beginSessionParams.replayMode = CUPTI_KernelReplay;
beginSessionParams.maxRangesPerPass = 1;
beginSessionParams.maxLaunchesPerPass = 1;

NV_CUPTI_CALL(cuptiProfilerBeginSession(&beginSessionParams));

// Set up the metric to be collected (e.g., "smsp__warps_launched.avg+")
CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
setConfigParams.pConfig = metricConfig;
setConfigParams.configSize = configSize;
setConfigParams.passIndex = 0;
NV_CUPTI_CALL(cuptiProfilerSetConfig(&setConfigParams));
```

### 4. Enabling Profiling and Running Kernels

Now we enable profiling and run our kernels:

```cpp
// Enable profiling
CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
    CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
NV_CUPTI_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));

// Launch the first kernel (VecAdd)
VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
cudaDeviceSynchronize();

// Launch the second kernel (VecSub)
VecSub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, N);
cudaDeviceSynchronize();

// Disable profiling
CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
    CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
NV_CUPTI_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
```

The autorange feature automatically detects these kernel launches and collects metrics for each one.

### 5. Processing the Results

After profiling, we need to process the collected data:

```cpp
// End the profiling session
CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
NV_CUPTI_CALL(cuptiProfilerEndSession(&endSessionParams));

// Unset the profiler configuration
CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
NV_CUPTI_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

// Get the number of ranges (kernels) that were profiled
CUpti_Profiler_CounterDataImage_GetNumRanges_Params getNumRangesParams = {
    CUpti_Profiler_CounterDataImage_GetNumRanges_Params_STRUCT_SIZE};
getNumRangesParams.pCounterDataImage = counterDataImage;
getNumRangesParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
NV_CUPTI_CALL(cuptiProfilerCounterDataImageGetNumRanges(&getNumRangesParams));

// Process each range (kernel)
for (int rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; rangeIndex++) {
    // Get the range name (kernel name)
    CUpti_Profiler_CounterDataImage_GetRangeName_Params getRangeNameParams = {
        CUpti_Profiler_CounterDataImage_GetRangeName_Params_STRUCT_SIZE};
    getRangeNameParams.pCounterDataImage = counterDataImage;
    getRangeNameParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    getRangeNameParams.rangeIndex = rangeIndex;
    NV_CUPTI_CALL(cuptiProfilerCounterDataImageGetRangeName(&getRangeNameParams));
    
    // Evaluate the metrics for this range
    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = {
        NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE};
    setCounterDataParams.pMetricsContext = metricsContext;
    setCounterDataParams.pCounterDataImage = counterDataImage;
    setCounterDataParams.rangeIndex = rangeIndex;
    NVPW_MetricsContext_SetCounterData(&setCounterDataParams);
    
    // Get the metric value
    double metricValue;
    NVPW_MetricsContext_EvaluateToGpuValues_Params evaluateToGpuParams = {
        NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE};
    evaluateToGpuParams.pMetricsContext = metricsContext;
    evaluateToGpuParams.metricNameBegin = metricName;
    evaluateToGpuParams.metricNameEnd = metricName + strlen(metricName);
    evaluateToGpuParams.pMetricValues = &metricValue;
    NVPW_MetricsContext_EvaluateToGpuValues(&evaluateToGpuParams);
    
    // Print the results
    printf("Range %d : %s\n  %s: %.1f\n", 
           rangeIndex, getRangeNameParams.pRangeName, metricName, metricValue);
}
```

## Sample Kernels

The sample includes two simple kernels to demonstrate profiling multiple functions:

```cpp
// Vector addition kernel
__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Vector subtraction kernel
__global__ void VecSub(const int* A, const int* B, int* D, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        D[i] = A[i] - B[i];
}
```

## Running the Tutorial

1. Make sure you have the required dependencies:
   ```bash
   cd ../  # Go to cupti_samples root
   ./install.sh
   ```

2. Build the sample:
   ```bash
   cd autorange_profiling
   make
   ```

3. Run the sample:
   ```bash
   ./autoRangeSample
   ```

## Understanding the Output

The sample produces output similar to:

```
Launching kernel: blocks 196, thread/block 256
Range 0 : VecAdd
  smsp__warps_launched.avg: 196.0
Range 1 : VecSub
  smsp__warps_launched.avg: 196.0
```

This shows:
1. The kernel configuration (blocks and threads per block)
2. Each detected kernel range with its name
3. The metric value for each kernel

In this example, both kernels launched the same number of warps (196), which makes sense since they have identical thread configurations and similar complexity.

## Available Metrics

The default metric `smsp__warps_launched.avg` shows the average number of warps launched per streaming multiprocessor (SM). Other useful metrics include:

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`: SM throughput as a percentage of peak
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`: Memory throughput as a percentage of peak
- `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed`: ALU utilization
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`: Global memory load operations
- `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`: Global memory store operations

To use a different metric, modify the `METRIC_NAME` definition in the code.

## Advanced Usage

### Collecting Multiple Metrics

To collect multiple metrics in a single run, you can:

1. Create a metric group in your configuration
2. Add multiple metrics to that group
3. Process each metric separately in the results phase

### Filtering Specific Kernels

While autorange profiles all kernels, you can filter the results by:

1. Checking the kernel name in the results processing phase
2. Only evaluating metrics for kernels you're interested in

## Performance Considerations

1. **Overhead**: Profiling adds overhead to kernel execution, especially when collecting complex metrics
2. **Multiple Passes**: Some metrics require multiple kernel replays, increasing execution time
3. **Memory Usage**: Counter data images can be large for applications with many kernels

## Next Steps

- Try collecting different metrics to understand various aspects of your kernels
- Apply autorange profiling to your own CUDA applications
- Compare the performance of different implementations of the same algorithm
- Use the collected metrics to identify optimization opportunities 