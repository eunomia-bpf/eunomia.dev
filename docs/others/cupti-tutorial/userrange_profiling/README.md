# CUPTI User-Range Profiling Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

When profiling CUDA applications, you often need to focus on specific sections of code rather than entire kernels or the complete application. CUPTI's User-Range Profiling feature allows you to define custom ranges in your code and collect performance metrics only within these ranges. This gives you precise control over which parts of your application are profiled, making it easier to identify and optimize performance bottlenecks in complex applications. This tutorial demonstrates how to use CUPTI's Profiler API to define custom ranges and collect performance metrics within them.

## What You'll Learn

- How to define and instrument user-specified ranges in your CUDA code
- Setting up the CUPTI Profiler API for user-range profiling
- Collecting GPU performance metrics within your defined ranges
- Processing and analyzing the collected metrics for each range
- Comparing performance across different parts of your application

## Understanding User-Range Profiling

Unlike automatic profiling that targets individual CUDA API calls or kernels, user-range profiling lets you:

1. Define logical sections of your code that may include multiple CUDA operations
2. Give meaningful names to these sections for easier analysis
3. Focus profiling resources on the most important parts of your application
4. Compare performance metrics across different algorithmic approaches

This approach is particularly useful when:
- Your application has distinct phases with different performance characteristics
- You want to compare different implementations of the same algorithm
- You need to profile a specific sequence of operations as a single unit
- You're optimizing a particular section of a larger application

## Code Walkthrough

### 1. Setting Up the CUPTI Profiler

First, we need to initialize the CUPTI Profiler API and configure it for user-range profiling:

```cpp
CUpti_Profiler_Initialize_Params initializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
CUPTI_API_CALL(cuptiProfilerInitialize(&initializeParams));

// Get the chip name for the current device
CUpti_Device_GetChipName_Params getChipNameParams = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
getChipNameParams.deviceIndex = 0;
CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
const char *chipName = getChipNameParams.pChipName;

// Create the metric configuration
NVPW_InitializeHost_Params initializeHostParams = {NVPW_InitializeHost_Params_STRUCT_SIZE};
NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

// Set up the metrics to collect
const char *metricNames[] = {"smsp__warps_launched.avg"};
struct MetricNameList metricList;
metricList.numMetrics = 1;
metricList.metricNames = metricNames;

// Create the counter data image and configuration
CUpti_Profiler_CounterDataImageOptions counterDataImageOptions = {
    CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE};
counterDataImageOptions.pChipName = chipName;
counterDataImageOptions.counterDataImageSize = 0;
CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&counterDataImageOptions));
counterDataImage = (uint8_t *)malloc(counterDataImageOptions.counterDataImageSize);
counterDataImageOptions.pCounterDataImage = counterDataImage;
CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&counterDataImageOptions));

// Create the counter data configuration
CUpti_Profiler_CounterDataImageCalculateScratchBufferSize_Params scratchBufferSizeParams = {
    CUpti_Profiler_CounterDataImageCalculateScratchBufferSize_Params_STRUCT_SIZE};
scratchBufferSizeParams.pOptions = &counterDataImageOptions;
CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));

// Configure the profiler for user-range profiling
CUpti_Profiler_UserRange_Config_Params configParams = {
    CUpti_Profiler_UserRange_Config_Params_STRUCT_SIZE};
configParams.pCounterDataPrefixImage = counterDataImage;
configParams.counterDataPrefixImageSize = counterDataImageOptions.counterDataImageSize;
configParams.maxRangesPerPass = 1;
configParams.maxLaunchesPerPass = 1;
CUPTI_API_CALL(cuptiProfilerUserRangeConfigureScratchBuffer(&configParams));
```

This code:
1. Initializes the CUPTI Profiler API
2. Gets the chip name for the current device
3. Sets up the metrics we want to collect (in this case, "smsp__warps_launched.avg")
4. Creates and configures the counter data image
5. Configures the profiler specifically for user-range profiling

### 2. Defining User Ranges in Your Code

Next, we define ranges around the code sections we want to profile:

```cpp
void profileVectorOperations(int *d_A, int *d_B, int *d_C, int numElements)
{
    // Define the range name
    const char *rangeName = "Vector Add-Subtract";
    
    // Start a user range
    CUpti_Profiler_BeginRange_Params beginRangeParams = {CUpti_Profiler_BeginRange_Params_STRUCT_SIZE};
    beginRangeParams.pRangeName = rangeName;
    CUPTI_API_CALL(cuptiProfilerBeginRange(&beginRangeParams));
    
    // Execute operations within the range
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel: blocks %d, thread/block %d\n", blocksPerGrid, threadsPerBlock);
    
    // First vector operation: Add
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    // Second vector operation: Subtract
    VecSub<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_B, d_C, numElements);
    
    // End the user range
    CUpti_Profiler_EndRange_Params endRangeParams = {CUpti_Profiler_EndRange_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEndRange(&endRangeParams));
}
```

This function:
1. Defines a meaningful name for the range ("Vector Add-Subtract")
2. Starts a user range with `cuptiProfilerBeginRange`
3. Executes multiple GPU operations within the range (two kernel launches)
4. Ends the range with `cuptiProfilerEndRange`

### 3. Starting and Stopping the Profiler Session

We need to start the profiler before our ranges and stop it after:

```cpp
int main(int argc, char *argv[])
{
    // Initialize CUDA and CUPTI
    initializeCuda();
    initializeProfiler();
    
    // Allocate and initialize data
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    // ... allocation and initialization code ...
    
    // Start the profiler session
    CUpti_Profiler_BeginSession_Params beginSessionParams = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    beginSessionParams.counterDataImageSize = counterDataImageOptions.counterDataImageSize;
    beginSessionParams.pCounterDataImage = counterDataImage;
    beginSessionParams.maxRangesPerPass = 1;
    beginSessionParams.maxLaunchesPerPass = 1;
    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));
    
    // Enable profiling
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
        CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
    
    // Execute our profiled code with user ranges
    profileVectorOperations(d_A, d_B, d_C, numElements);
    
    // Disable profiling
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
        CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
    
    // End the profiler session
    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
    
    // Process the results
    processCounterData();
    
    // Clean up
    // ... cleanup code ...
    
    return 0;
}
```

This code:
1. Initializes CUDA and the CUPTI profiler
2. Starts a profiler session with `cuptiProfilerBeginSession`
3. Enables profiling with `cuptiProfilerEnableProfiling`
4. Executes the code containing our user ranges
5. Disables profiling and ends the session
6. Processes the collected data

### 4. Processing the Collected Metrics

After profiling, we need to process the counter data to calculate metrics:

```cpp
void processCounterData()
{
    // Get the chip name for the current device
    CUpti_Device_GetChipName_Params getChipNameParams = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    getChipNameParams.deviceIndex = 0;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    const char *chipName = getChipNameParams.pChipName;
    
    // Set up the metric evaluation request
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParams = {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    calculateScratchBufferSizeParams.pChipName = chipName;
    NVPW_API_CALL(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calculateScratchBufferSizeParams));
    
    // Create the metrics evaluator
    NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {
        NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    metricEvaluatorInitializeParams.pChipName = chipName;
    metricEvaluatorInitializeParams.scratchBufferSize = calculateScratchBufferSizeParams.scratchBufferSize;
    metricEvaluatorInitializeParams.pScratchBuffer = malloc(calculateScratchBufferSizeParams.scratchBufferSize);
    NVPW_API_CALL(NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams));
    
    // Get the number of ranges
    CUpti_Profiler_GetNumRanges_Params getNumRangesParams = {CUpti_Profiler_GetNumRanges_Params_STRUCT_SIZE};
    getNumRangesParams.pCounterDataImage = counterDataImage;
    CUPTI_API_CALL(cuptiProfilerGetNumRanges(&getNumRangesParams));
    
    // For each range, get the metrics
    for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex) {
        // Get the range name
        CUpti_Profiler_GetRangeName_Params getRangeNameParams = {CUpti_Profiler_GetRangeName_Params_STRUCT_SIZE};
        getRangeNameParams.pCounterDataImage = counterDataImage;
        getRangeNameParams.rangeIndex = rangeIndex;
        CUPTI_API_CALL(cuptiProfilerGetRangeName(&getRangeNameParams));
        
        printf("Range %zu : %s\n", rangeIndex, getRangeNameParams.pRangeName);
        
        // Evaluate the metrics for this range
        NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttribParams = {
            NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE};
        setDeviceAttribParams.pMetricsEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;
        setDeviceAttribParams.pCounterDataImage = counterDataImage;
        setDeviceAttribParams.counterDataImageSize = counterDataImageOptions.counterDataImageSize;
        setDeviceAttribParams.rangeIndex = rangeIndex;
        NVPW_API_CALL(NVPW_MetricsEvaluator_SetDeviceAttributes(&setDeviceAttribParams));
        
        // For each metric, get and display its value
        for (size_t metricIndex = 0; metricIndex < metricList.numMetrics; ++metricIndex) {
            const char *metricName = metricList.metricNames[metricIndex];
            
            NVPW_MetricsEvaluator_GetMetricValue_Params getMetricValueParams = {
                NVPW_MetricsEvaluator_GetMetricValue_Params_STRUCT_SIZE};
            getMetricValueParams.pMetricsEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;
            getMetricValueParams.metricName = metricName;
            NVPW_API_CALL(NVPW_MetricsEvaluator_GetMetricValue(&getMetricValueParams));
            
            printf("  %s: %.1f\n", metricName, getMetricValueParams.metricValue);
        }
    }
    
    // Clean up
    free(metricEvaluatorInitializeParams.pScratchBuffer);
}
```

This function:
1. Sets up a metrics evaluator for the current device
2. Gets the number of ranges that were profiled
3. For each range:
   - Gets the range name
   - Sets up the device attributes for metric evaluation
   - Calculates and displays the value of each requested metric

### 5. Sample Vector Operation Kernels

Here are the simple vector operation kernels we're profiling:

```cpp
__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

__global__ void VecSub(const int* A, const int* B, int* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] - B[i];
    }
}
```

These kernels:
1. Calculate the global thread index
2. Check if the index is within bounds
3. Perform a simple vector addition or subtraction

## Running the Tutorial

1. Build the sample:
   ```bash
   make
   ```

2. Run the user-range profiling example:
   ```bash
   ./userRangeSample
   ```

## Understanding the Output

When you run the user-range profiling example, you'll see output similar to this:

```
Launching kernel: blocks 196, thread/block 256
Range 0 : Vector Add-Subtract
  smsp__warps_launched.avg: 392.0
```

Let's analyze this output:

1. **Kernel Launch Information**:
   - `blocks 196, thread/block 256`: Shows the kernel configuration

2. **Range Information**:
   - `Range 0 : Vector Add-Subtract`: Identifies the user-defined range by index and name

3. **Metric Values**:
   - `smsp__warps_launched.avg: 392.0`: Shows the average number of warps launched per streaming multiprocessor

The metric `smsp__warps_launched.avg` indicates the average number of warps launched per streaming multiprocessor (SM) during the execution of operations within our range. In this case, the value is 392.0, which means that on average, each SM launched 392 warps during the vector operations.

## Advanced User-Range Profiling Techniques

### 1. Nested Ranges

You can create nested ranges to profile hierarchical code structures:

```cpp
// Start outer range
cuptiProfilerBeginRange(&outerRangeParams);

// First operation
operation1();

// Start inner range
cuptiProfilerBeginRange(&innerRangeParams);
operation2();
operation3();
// End inner range
cuptiProfilerEndRange(&innerEndRangeParams);

operation4();

// End outer range
cuptiProfilerEndRange(&outerEndRangeParams);
```

### 2. Multiple Metrics

You can collect multiple metrics in a single profiling session:

```cpp
const char *metricNames[] = {
    "smsp__warps_launched.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum"
};
metricList.numMetrics = 4;
metricList.metricNames = metricNames;
```

### 3. Comparing Implementations

You can use user ranges to compare different implementations of the same algorithm:

```cpp
// Implementation 1
cuptiProfilerBeginRange(&impl1RangeParams);
algorithmImplementation1();
cuptiProfilerEndRange(&impl1EndRangeParams);

// Implementation 2
cuptiProfilerBeginRange(&impl2RangeParams);
algorithmImplementation2();
cuptiProfilerEndRange(&impl2EndRangeParams);
```

### 4. Profiling Specific Application Phases

You can profile specific phases of your application:

```cpp
// Data preparation phase
cuptiProfilerBeginRange(&prepRangeParams);
prepareData();
cuptiProfilerEndRange(&prepEndRangeParams);

// Computation phase
cuptiProfilerBeginRange(&computeRangeParams);
performComputation();
cuptiProfilerEndRange(&computeEndRangeParams);

// Result processing phase
cuptiProfilerBeginRange(&processRangeParams);
processResults();
cuptiProfilerEndRange(&processEndRangeParams);
```

## Common Metrics for User-Range Profiling

Here are some useful metrics to collect in user ranges:

1. **Compute Utilization**:
   - `sm__warps_active.avg.pct_of_peak_sustained_active`: Percentage of warps active
   - `smsp__thread_inst_executed.avg.pct_of_peak_sustained_active`: Thread instruction execution efficiency

2. **Memory Performance**:
   - `dram__bytes_read.sum`: Total bytes read from DRAM
   - `dram__bytes_write.sum`: Total bytes written to DRAM
   - `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`: Global memory load transactions
   - `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`: Global memory store transactions

3. **Execution Statistics**:
   - `smsp__warps_launched.avg`: Average warps launched per SM
   - `smsp__cycles_active.avg`: Average cycles with at least one active warp
   - `sm__cycles_elapsed.avg`: Average cycles elapsed

4. **Instruction Mix**:
   - `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`: Float add instructions
   - `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum`: Float multiply instructions
   - `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum`: Fused multiply-add instructions

## Next Steps

- Apply user-range profiling to your own CUDA applications
- Experiment with different metrics to gain insights into performance
- Use nested ranges to analyze hierarchical code structures
- Compare different implementations of algorithms using user ranges
- Combine user-range profiling with other CUPTI features for comprehensive analysis 