# CUPTI Program Counter (PC) Sampling Continuous Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

Program Counter (PC) sampling is a powerful profiling technique that allows you to understand where your CUDA kernels spend their execution time at the assembly instruction level. This tutorial demonstrates how to implement continuous PC sampling that can monitor any CUDA application without requiring source code modifications.

## What You'll Learn

- How to build a dynamic library for PC sampling injection
- Techniques for continuous profiling of CUDA applications
- Understanding PC sampling data and stall reasons
- Cross-platform implementation (Linux and Windows)
- Using the profiling library with existing applications

## Understanding PC Sampling Continuous

PC sampling continuous differs from other profiling methods because it:

1. **Operates at the assembly level**: Provides insights into actual GPU instruction execution
2. **Requires no source modifications**: Can profile any CUDA application
3. **Works via library injection**: Uses dynamic loading to intercept CUDA calls
4. **Provides stall reason analysis**: Shows why warps are not making progress
5. **Supports real-time monitoring**: Can observe performance during execution

## Architecture Overview

The continuous PC sampling system consists of:

1. **Dynamic Library**: `libpc_sampling_continuous.so` (Linux) or `pc_sampling_continuous.lib` (Windows)
2. **Injection Mechanism**: Uses `LD_PRELOAD` (Linux) or DLL injection (Windows)
3. **CUPTI Integration**: Leverages CUPTI's PC sampling APIs
4. **Helper Script**: `libpc_sampling_continuous.pl` for easy execution

## Detailed Component Analysis

### 1. Helper Script (`libpc_sampling_continuous.pl`)

The Perl script serves as a wrapper that simplifies the PC sampling process. Here's how it works:

#### Script Workflow

1. **Command Line Parsing (Lines 29-41)**
   ```perl
   GetOptions( 'help'                => \$help
             , 'app=s'               => \$applicationName
             , 'collection-mode=i'   => \$collectionMode
             , 'sampling-period=i'   => \$samplingPeriod
             # ... more options
   ```
   - Uses Perl's `Getopt::Long` module to parse command-line arguments
   - Supports various sampling configuration parameters

2. **Parameter Validation (Lines 44-104)**
   - **Collection Mode**: Validates values (1 for continuous, 2 for kernel-serialized)
   - **Sampling Period**: Ensures value is between 5-31 (represents 2^n cycles)
   - **Buffer Sizes**: Validates scratch buffer and hardware buffer sizes
   - Builds command line options string for passing to the injection library

3. **Library Path Verification (`init` function, Lines 150-233)**
   ```perl
   sub init {
       my $ldLibraryPath = $ENV{'LD_LIBRARY_PATH'};
       my @libPaths = split /:/, $ldLibraryPath;
   ```
   - Checks for required libraries in the system paths:
     - `libpc_sampling_continuous.so`: The main profiling library
     - `libcupti.so`: CUPTI library for GPU profiling
     - `libpcsamplingutil.so`: Utility library for PC sampling
   - Sets `CUDA_INJECTION64_PATH` environment variable for CUDA injection

4. **Application Execution (`RunApplication` function, Lines 235-244)**
   ```perl
   sub RunApplication {
       $ENV{INJECTION_PARAM} = $injectionParameters;
       my $returnCode = system($applicationName);
   }
   ```
   - Sets injection parameters as environment variable
   - Launches the target application with the injection library loaded

#### Key Configuration Parameters

| Parameter | Description | Default | Valid Range |
|-----------|-------------|---------|-------------|
| `--collection-mode` | Sampling mode (1=continuous, 2=serialized) | 1 | 1-2 |
| `--sampling-period` | Sets sampling to 2^n cycles | - | 5-31 |
| `--scratch-buf-size` | Buffer for temporary PC records | 1 MB | Any size |
| `--hw-buf-size` | Hardware buffer size | 512 MB | Any size |
| `--pc-config-buf-record-count` | PC records for configuration | 5000 | Any count |
| `--pc-circular-buf-record-count` | Records per circular buffer | 500 | Any count |
| `--circular-buf-count` | Number of circular buffers | 10 | Any count |
| `--file-name` | Output filename | pcsampling.dat | Any name |

### 2. Core Implementation (`pc_sampling_continuous.cpp`)

The C++ implementation handles the actual PC sampling through CUPTI callbacks and data collection.

#### Initialization Flow

1. **Entry Point (`InitializeInjection`, Lines 987-1025)**
   ```cpp
   extern "C" int InitializeInjection(void) {
       // Read environment parameters
       ReadInputParams();
       
       // Subscribe to CUPTI callbacks
       CUPTI_API_CALL(cuptiSubscribe(&subscriber, 
                      (CUpti_CallbackFunc)&CallbackHandler, NULL));
       
       // Enable callbacks for all kernel launch variants
       CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, 
                      CUPTI_CB_DOMAIN_DRIVER_API, 
                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
       // ... more launch callbacks
       
       // Enable resource callbacks
       CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, 
                      CUPTI_CB_DOMAIN_RESOURCE, 
                      CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
   }
   ```
   - Called automatically when CUDA loads the injection library
   - Subscribes to all kernel launch callbacks and resource events
   - Sets up exit handler for cleanup

#### Data Structures

2. **Context Information Management (Lines 95-106)**
   ```cpp
   typedef struct ContextInfo_st {
       uint32_t contextUid;
       CUpti_PCSamplingData pcSamplingData;
       std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingConfigurationInfo;
       PcSamplingStallReasons pcSamplingStallReasons;
       bool ctxDestroyed;
       uint8_t *pPcSamplingBuffer;
   } ContextInfo;
   ```
   - Stores per-context PC sampling configuration
   - Maintains sampling data and stall reason information
   - Tracks context lifecycle

#### Callback System

3. **Main Callback Handler (`CallbackHandler`, Lines 758-983)**
   ```cpp
   void CallbackHandler(void *pUserdata, 
                       CUpti_CallbackDomain domain,
                       CUpti_CallbackId callbackId, 
                       void *pCallbackData) {
       switch (domain) {
           case CUPTI_CB_DOMAIN_DRIVER_API:
               // Handle kernel launches
               HandleDriverApiCallback(callbackId, pCallbackData);
               break;
           case CUPTI_CB_DOMAIN_RESOURCE:
               // Handle context and module events
               HandleResourceCallback(callbackId, pCallbackData);
               break;
       }
   }
   ```

4. **Context Creation Handling**
   - Enables PC sampling for new contexts
   - Configures sampling parameters (stall reasons, buffer sizes)
   - Creates worker thread for data processing (first context only)
   - Allocates circular buffers for data collection

5. **Kernel Launch Processing**
   - **Serialized Mode**: Flushes all PC records after each kernel
   - **Continuous Mode**: Flushes when buffer reaches threshold
   - Uses `cuptiPCSamplingGetData()` to retrieve samples
   - Pushes data to queue for file writing

6. **Module Load Events**
   - Handles dynamic module loading/unloading
   - Flushes any pending PC records when modules change
   - Ensures data consistency across module boundaries

#### Data Collection Workflow

7. **PC Sampling Data Retrieval**
   ```cpp
   bool GetPcSamplingDataFromCupti(
       CUpti_PCSamplingGetDataParams &params,
       ContextInfo *pContextInfo) {
       // Allocate circular buffer
       params.pPcData = g_circularBuffer[g_bufferIndexForCupti];
       params.pcDataBufferSize = g_circularbufSize;
       
       // Get data from CUPTI
       CUptiResult result = cuptiPCSamplingGetData(&params);
       
       // Handle buffer and queue data for writing
       if (pContextInfo->pcSamplingData.totalNumPcs > 0) {
           g_pcSampDataQueue.push(
               std::make_pair(&pContextInfo->pcSamplingData, 
                            pContextInfo));
       }
   }
   ```

8. **Worker Thread (`StoreDataInFile`, Lines 541-583)**
   ```cpp
   void StoreDataInFile() {
       while (g_running || !g_pcSampDataQueue.empty()) {
           if (!g_pcSampDataQueue.empty()) {
               // Get data from queue
               auto pcSampData = g_pcSampDataQueue.front();
               g_pcSampDataQueue.pop();
               
               // Write to file using CUPTI utility
               CuptiUtilPutPcSampData(fileName, 
                                    &pContextInfo->pcSamplingStallReasons,
                                    &pcSamplingConfigurationInfo,
                                    &pcSamplingData);
           }
       }
   }
   ```
   - Runs continuously in background
   - Processes queued PC sampling data
   - Writes data to binary files using CUPTI utilities
   - Creates per-context output files

#### Cleanup and Exit

9. **Exit Handler (`AtExitHandler`, Lines 595-673)**
   ```cpp
   void AtExitHandler() {
       // Disable PC sampling for all active contexts
       for (auto& itr: g_contextInfoMap) {
           // Flush remaining data
           while (itr.second->pcSamplingData.remainingNumPcs > 0) {
               GetPcSamplingData(pcSamplingGetDataParams, itr.second);
           }
           
           // Disable sampling
           cuptiPCSamplingDisable(&pcSamplingDisableParams);
           
           // Queue final buffer for writing
           g_pcSampDataQueue.push(
               std::make_pair(&itr.second->pcSamplingData, 
                            itr.second));
       }
       
       // Join worker thread and cleanup
       g_thread.join();
       FreeAllocatedMemory();
   }
   ```

### 3. Data Flow Architecture

```
Application Launch
        |
        v
[libpc_sampling_continuous.pl]
        |
        | Sets CUDA_INJECTION64_PATH
        | Sets INJECTION_PARAM
        v
[CUDA Runtime Loads Library]
        |
        v
[InitializeInjection()]
        |
        | Subscribes to callbacks
        v
[Context Created] -----> [Configure PC Sampling]
        |                         |
        v                         v
[Kernel Launch] -----> [Collect PC Samples]
        |                         |
        v                         v
[Module Events] -----> [Flush to Circular Buffer]
        |                         |
        v                         v
[Worker Thread] <------ [Queue PC Data]
        |
        v
[Write to File]
        |
        v
[N_pcsampling.dat]
```

## Building the Sample

### Linux Build Process

1. Navigate to the pc_sampling_continuous directory:
   ```bash
   cd pc_sampling_continuous
   ```

2. Build using the provided Makefile:
   ```bash
   make
   ```
   
   This creates `libpc_sampling_continuous.so` in the current directory.

### Windows Build Process

For Windows, you need to build the Microsoft Detours library first:

1. **Download Detours source**:
   - From GitHub: https://github.com/microsoft/Detours
   - Or Microsoft: https://www.microsoft.com/en-us/download/details.aspx?id=52586

2. **Build Detours**:
   ```cmd
   # Extract and navigate to Detours folder
   set DETOURS_TARGET_PROCESSOR=X64
   "Program Files (x86)\Microsoft Visual Studio\<version>\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
   NMAKE
   ```

3. **Copy required files**:
   ```cmd
   copy detours.h <pc_sampling_continuous_folder>
   copy detours.lib <pc_sampling_continuous_folder>
   ```

4. **Build the sample**:
   ```cmd
   nmake
   ```
   
   This creates `pc_sampling_continuous.lib`.

## Running the Sample

### Linux Execution

1. **Set up library paths**:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/CUPTI/lib64:/path/to/pc_sampling_continuous:/path/to/pcsamplingutil
   ```

   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-13.0/extras/CUPTI/lib64:/usr/local/cuda-13.0/extras/CUPTI/samples/pc_sampling_continuous:/usr/local/cuda-13.0/extras/CUPTI/samples/pc_sampling_utility
   ```

2. **Use the helper script**:
   ```bash
   ./libpc_sampling_continuous.pl --help
   ```
   
   This shows all available options.

3. **Run with your application**:
   ```bash
   ./libpc_sampling_continuous.pl --app /path/to/your/cuda/application
   ```

### Windows Execution

1. **Set up library paths**:
   ```cmd
   set PATH=%PATH%;C:\path\to\CUPTI\bin;C:\path\to\pc_sampling_continuous;C:\path\to\pcsamplingutil
   ```

2. **Run with your application**:
   ```cmd
   pc_sampling_continuous.exe your_cuda_application.exe
   ```

## Understanding the Output

### Binary Data Format

The output files (`N_pcsampling.dat`) contain binary data in CUPTI's PC sampling format:

1. **Header Section**
   - Magic string: "CUPS" (CUPTI Unified Profiling Samples)
   - Version information
   - Number of buffers
   - Configuration parameters

2. **Configuration Data**
   - Stall reason names and indices
   - Sampling period and collection mode
   - Buffer sizes and counts

3. **PC Sample Records**
   - Program counter addresses
   - Stall reason codes
   - Sample counts
   - Timestamp information

### Parsing the Data

Use the `pc_sampling_utility` tool to parse and analyze the binary data:

```bash
# Basic parsing
../pc_sampling_utility/pc_sampling_utility --file-name 2_pcsampling.dat

# With source correlation
../pc_sampling_utility/pc_sampling_utility --file-name 2_pcsampling.dat --disable-source-correlation

# Verbose output
../pc_sampling_utility/pc_sampling_utility --file-name 2_pcsampling.dat --verbose
```

### Example Parsed Output

```
Function: vectorAdd
Module: module_1
Total Samples: 10000

PC Address    | Stall Reason           | Count | Percentage
0x7f8b2c1000 | MEMORY_DEPENDENCY      | 3500  | 35.0%
0x7f8b2c1008 | EXECUTION_DEPENDENCY   | 2000  | 20.0%
0x7f8b2c1010 | NOT_SELECTED          | 1500  | 15.0%
0x7f8b2c1018 | SYNCHRONIZATION       | 1000  | 10.0%
...
```

## Advanced Configuration

### Performance Tuning

1. **Sampling Period Optimization**
   - Lower values (5-10): High detail, more overhead
   - Medium values (11-20): Balanced approach
   - Higher values (21-31): Low overhead, less detail

2. **Buffer Size Considerations**
   ```
   Scratch Buffer Size = (Expected PCs) × (16 bytes + 16 bytes × stall_reasons)
   
   Example: 1000 PCs with 4 stall reasons
   = 1000 × (16 + 16 × 4)
   = 1000 × 80 bytes
   = 80 KB minimum
   ```

3. **Collection Mode Selection**
   - **Continuous**: Best for long-running kernels
   - **Serialized**: Better for short, frequent kernel launches

### Memory Management

The system uses a multi-buffer approach:

1. **Configuration Buffer**: Holds PC sampling setup (default: 5000 records)
2. **Circular Buffers**: Temporary storage for collected data (default: 10 buffers × 500 records)
3. **Hardware Buffer**: GPU-side storage (default: 512 MB)
4. **Scratch Buffer**: Working memory for data processing (default: 1 MB)

### Stall Reason Analysis

Common stall reasons and their implications:

| Stall Reason | Description | Optimization Strategy |
|--------------|-------------|----------------------|
| MEMORY_DEPENDENCY | Waiting for memory operations | Improve memory coalescing, use shared memory |
| EXECUTION_DEPENDENCY | Waiting for previous instructions | Reorder instructions, increase ILP |
| NOT_SELECTED | Warp not scheduled | Balance workload, reduce divergence |
| SYNCHRONIZATION | Waiting at sync points | Minimize __syncthreads(), optimize barriers |
| TEXTURE | Waiting for texture fetch | Optimize texture cache usage |
| CONSTANT_MEMORY | Waiting for constant memory | Use shared memory for frequently accessed constants |

## Troubleshooting

### Common Issues and Solutions

1. **Library Loading Failures**
   ```
   ERROR: Library libpc_sampling_continuous.so not present
   ```
   Solution: Ensure LD_LIBRARY_PATH includes the library directory

2. **CUPTI Initialization Errors**
   ```
   CUPTI_ERROR_NOT_INITIALIZED
   ```
   Solution: Verify CUDA and CUPTI versions match

3. **Buffer Overflow**
   ```
   WARNING: N records are discarded during cuptiPCSamplingDisable()
   ```
   Solution: Increase buffer sizes using command-line parameters

4. **Permission Denied**
   ```
   Failed to open file for writing
   ```
   Solution: Ensure write permissions in output directory

### Debug Tips

1. **Enable Verbose Logging**
   ```bash
   ./libpc_sampling_continuous.pl --app ./myapp --verbose
   ```

2. **Check Library Dependencies**
   ```bash
   ldd libpc_sampling_continuous.so
   ```

3. **Monitor System Resources**
   ```bash
   # Check available memory
   free -h
   
   # Monitor during execution
   watch -n 1 'free -h'
   ```

4. **Test with Simple Kernels First**
   ```cuda
   __global__ void simpleKernel() {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       // Simple operation for testing
   }
   ```

## Performance Impact

PC sampling overhead depends on:

1. **Sampling Frequency**: Higher frequency = more overhead
2. **Kernel Duration**: Longer kernels amortize setup costs
3. **Buffer Sizes**: Larger buffers reduce flush frequency
4. **Collection Mode**: Continuous mode has lower overhead than serialized

Typical overhead ranges:
- Low sampling (period=20-31): 1-5% overhead
- Medium sampling (period=10-19): 5-15% overhead
- High sampling (period=5-9): 15-30% overhead

## Best Practices

1. **Start with Conservative Settings**
   - Use larger sampling periods initially
   - Increase detail gradually as needed

2. **Profile Representative Workloads**
   - Ensure profiling covers typical use cases
   - Run multiple iterations for statistical significance

3. **Correlate with Other Metrics**
   - Combine with nvprof/ncu metrics
   - Cross-reference with application-level timing

4. **Automate Analysis**
   - Script data parsing and analysis
   - Create performance regression tests

## Next Steps

- Experiment with different sampling frequencies to find optimal settings
- Apply continuous PC sampling to your own CUDA applications
- Combine with the `pc_sampling_utility` to analyze collected data
- Explore correlation with source code using debug symbols
- Integrate PC sampling data with your performance analysis workflow