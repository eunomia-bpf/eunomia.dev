# CUPTI PC Sampling Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

When optimizing CUDA kernels, it's essential to know which parts of your code consume the most execution time. Program Counter (PC) sampling is a powerful technique that allows you to identify hotspots in your GPU code with minimal performance overhead. This tutorial demonstrates how to use CUPTI's PC Sampling API to collect program counter samples during kernel execution and map them back to your source code, helping you focus your optimization efforts where they'll have the most impact.

## What You'll Learn

- How to configure and enable PC sampling for CUDA kernels
- Collecting and processing PC samples during kernel execution
- Mapping PC addresses to source code locations
- Identifying performance hotspots in your CUDA code
- Analyzing sampling data to guide optimization efforts

## Understanding PC Sampling
Program Counter (PC) sampling works by periodically recording the current instruction being executed by each active thread. The process works as follows:

1. The GPU hardware periodically samples the program counter of active warps
2. These samples are collected and buffered during kernel execution
3. After the kernel completes, the samples are analyzed to determine which code regions were executed most frequently
4. The PC addresses are mapped back to source code using debugging information

Unlike instrumentation-based profiling, PC sampling has minimal impact on kernel performance and provides a statistical view of where the GPU spends its execution time.

## Code Walkthrough

### 1. Setting Up PC Sampling

First, we need to configure and enable PC sampling:

```cpp
CUpti_PCSamplingConfig config;
memset(&config, 0, sizeof(config));
config.size = sizeof(config);
config.samplingPeriod = 5;  // Sample every 5 cycles
config.samplingPeriod2 = 0; // Not used in this example
config.samplingPeriodRatio = 0; // Not used in this example
config.collectingMode = CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL;
config.samplingBufferSize = 0x2000; // 8KB buffer
config.stoppingCount = 0; // No stopping condition

// Get the current CUDA context
CUcontext context;
DRIVER_API_CALL(cuCtxGetCurrent(&context));

// Enable PC sampling
CUPTI_CALL(cuptiPCSamplingEnable(&config));
```

This code:
1. Creates a configuration structure for PC sampling
2. Sets the sampling period (how often to sample the PC)
3. Configures the collection mode to sample during kernel execution
4. Sets the buffer size for storing samples
5. Enables PC sampling on the current CUDA context

### 2. Registering Buffer Callbacks

To handle the PC sampling data, we need to register callbacks that will be called when buffers are full:

```cpp
// Register callbacks for buffer handling
CUPTI_CALL(cuptiPCSamplingRegisterBufferHandler(handleBuffer, userData));

// Callback function to process PC sampling buffers
void handleBuffer(uint8_t *buffer, size_t size, size_t validSize, void *userData)
{
    PCData *pcData = (PCData *)userData;
    
    // Process all records in the buffer
    CUpti_PCSamplingData *record = (CUpti_PCSamplingData *)buffer;
    
    while (validSize >= sizeof(CUpti_PCSamplingData)) {
        // Process the PC sample
        processPCSample(record, pcData);
        
        // Move to the next record
        validSize -= sizeof(CUpti_PCSamplingData);
        record++;
    }
}
```

This code:
1. Registers a callback function that will be called when PC sampling buffers are ready
2. In the callback, processes each PC sampling record
3. Updates statistics based on the collected samples

### 3. Processing PC Samples

For each PC sample, we need to process and store the information:

```cpp
void processPCSample(CUpti_PCSamplingData *sample, PCData *pcData)
{
    // Extract information from the sample
    uint64_t pc = sample->pc;
    uint32_t functionId = sample->functionId;
    uint32_t stall = sample->stallReason;
    
    // Update the PC histogram
    pcData->totalSamples++;
    
    // Find or create an entry for this PC
    PCEntry *entry = findPCEntry(pcData, pc);
    if (!entry) {
        entry = createPCEntry(pcData, pc);
    }
    
    // Update the sample count for this PC
    entry->sampleCount++;
    
    // Update stall reason counts if needed
    if (stall != CUPTI_PC_SAMPLING_STALL_NONE) {
        entry->stallCount[stall]++;
    }
}
```

This function:
1. Extracts the program counter (PC) value from the sample
2. Updates the total sample count
3. Finds or creates an entry for this PC in our data structure
4. Updates statistics for this PC, including stall reasons

### 4. Mapping PCs to Source Code

To make the PC values meaningful, we need to map them back to source code:

```cpp
void mapPCsToSource(PCData *pcData)
{
    // Get the CUDA module for the kernel
    CUmodule module;
    DRIVER_API_CALL(cuModuleGetFunction(&module, pcData->function));
    
    // For each PC entry, get source information
    for (int i = 0; i < pcData->numEntries; i++) {
        PCEntry *entry = &pcData->entries[i];
        
        // Get source file and line information
        CUpti_LineInfo lineInfo;
        lineInfo.size = sizeof(lineInfo);
        
        CUPTI_CALL(cuptiGetLineInfo(module, entry->pc, &lineInfo));
        
        // Store source information
        if (lineInfo.lineInfoValid) {
            entry->fileName = strdup(lineInfo.fileName);
            entry->lineNumber = lineInfo.lineNumber;
            entry->functionName = strdup(lineInfo.functionName);
        } else {
            entry->fileName = strdup("unknown");
            entry->lineNumber = 0;
            entry->functionName = strdup("unknown");
        }
    }
}
```

This function:
1. Gets the CUDA module for our kernel
2. For each PC entry, calls CUPTI to get source line information
3. Stores the file name, line number, and function name for each PC

### 5. Sample Kernel with Hotspots

To demonstrate PC sampling, we'll use a kernel with some intentional hotspots:

```cpp
__global__ void sampleKernel(float *data, int size, int iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float value = data[idx];
        
        // Hotspot 1: Compute-intensive loop
        for (int i = 0; i < iterations; i++) {
            value = value * value + 0.5f;
        }
        
        // Hotspot 2: Conditional branch with divergence
        if (idx % 2 == 0) {
            for (int i = 0; i < iterations / 2; i++) {
                value = sinf(value);
            }
        } else {
            for (int i = 0; i < iterations / 4; i++) {
                value = cosf(value);
            }
        }
        
        // Hotspot 3: Memory access pattern
        int offset = (idx * 17) % size;
        value += data[offset];
        
        data[idx] = value;
    }
}
```

This kernel has several characteristics that will show up in PC sampling:
1. A compute-intensive loop that will be a major hotspot
2. Conditional branches that cause thread divergence
3. Non-coalesced memory access patterns

### 6. Analyzing and Displaying Results

After collecting samples, we analyze and display the results:

```cpp
void analyzeResults(PCData *pcData)
{
    // Sort PC entries by sample count
    qsort(pcData->entries, pcData->numEntries, sizeof(PCEntry), comparePCEntries);
    
    // Print summary
    printf("\nPC Sampling Results:\n");
    printf("  Total samples collected: %llu\n", pcData->totalSamples);
    printf("  Unique PC addresses: %d\n\n", pcData->numEntries);
    
    // Print top hotspots
    printf("Top 5 hotspots:\n");
    int numToShow = (pcData->numEntries < 5) ? pcData->numEntries : 5;
    
    for (int i = 0; i < numToShow; i++) {
        PCEntry *entry = &pcData->entries[i];
        float percentage = 100.0f * entry->sampleCount / pcData->totalSamples;
        
        printf("  %d. PC: 0x%llx (%.1f%% of samples) - %s:%d - %s\n",
               i + 1, entry->pc, percentage,
               entry->fileName, entry->lineNumber, entry->functionName);
        
        // Print stall reasons if available
        if (entry->stallCount[CUPTI_PC_SAMPLING_STALL_MEMORY_THROTTLE] > 0) {
            printf("     Memory throttle stalls: %d\n", 
                  entry->stallCount[CUPTI_PC_SAMPLING_STALL_MEMORY_THROTTLE]);
        }
        if (entry->stallCount[CUPTI_PC_SAMPLING_STALL_SYNC] > 0) {
            printf("     Synchronization stalls: %d\n", 
                  entry->stallCount[CUPTI_PC_SAMPLING_STALL_SYNC]);
        }
        // ... print other stall reasons ...
    }
}
```

This function:
1. Sorts PC entries by sample count to identify hotspots
2. Displays summary information about the collected samples
3. Shows the top hotspots with source code locations
4. Includes information about stall reasons for each hotspot

## Running the Tutorial

1. Build the sample:
   ```bash
   make
   ```

2. Run the PC sampling example:
   ```bash
   ./pc_sampling
   ```

## Understanding the Output

When you run the PC sampling example, you'll see output similar to this:

```
PC Sampling Configuration:
  Sample period: 5
  Collection mode: Kernel
  Sample buffer size: 8192 samples

Running test kernel...
Kernel execution complete.

PC Sampling Results:
  Total samples collected: 5432
  Unique PC addresses: 128

Top 5 hotspots:
  1. PC: 0x1a40 (25.3% of samples) - sampleKernel.cu:45 - Compute-intensive loop
     Memory throttle stalls: 0
     Instruction fetch stalls: 0
  2. PC: 0x1b00 (18.7% of samples) - sampleKernel.cu:58 - Conditional branch (sinf)
     Memory throttle stalls: 0
     Instruction fetch stalls: 12
  3. PC: 0x1c80 (12.1% of samples) - sampleKernel.cu:64 - Conditional branch (cosf)
     Memory throttle stalls: 0
     Instruction fetch stalls: 8
  4. PC: 0x1d20 (8.4% of samples)  - sampleKernel.cu:70 - Memory access pattern
     Memory throttle stalls: 145
     Instruction fetch stalls: 0
  5. PC: 0x1e60 (5.2% of samples)  - sampleKernel.cu:73 - Final store
     Memory throttle stalls: 78
     Instruction fetch stalls: 0

Performance insights:
  - 25.3% of execution time spent in the compute-intensive loop
  - 30.8% of execution time spent in conditional branches (sinf/cosf)
  - 13.6% of execution time spent in memory operations
  - Memory throttle stalls primarily occur during non-coalesced memory access
```

Let's analyze this output:

1. **Configuration Information**:
   - The sampling period (5 cycles)
   - The collection mode (during kernel execution)
   - The buffer size for storing samples

2. **Overall Statistics**:
   - Total number of samples collected (5432)
   - Number of unique PC addresses (128)

3. **Hotspot Analysis**:
   - The top 5 most frequently sampled PC addresses
   - For each hotspot:
     - The source file and line number
     - The percentage of total samples
     - Stall reasons (if any)

4. **Performance Insights**:
   - Summary of where the kernel is spending time
   - Identification of potential optimization targets

## Interpreting PC Sampling Results

### Identifying Compute Bottlenecks

A high percentage of samples in compute-intensive code indicates that the kernel is compute-bound. In our example, 25.3% of samples are in the compute-intensive loop. Optimization strategies include:

- Algorithm improvements
- Instruction-level optimizations
- Reducing arithmetic operations

### Detecting Thread Divergence

When samples are distributed across different code paths in conditional branches, it may indicate thread divergence. In our example, 30.8% of samples are in the two conditional branches. Optimization strategies include:

- Restructuring conditionals to reduce divergence
- Moving conditionals outside of loops
- Using predication instead of branching

### Finding Memory Access Issues

Samples with memory throttle stalls indicate memory access issues. In our example, the non-coalesced memory access pattern shows significant stalls. Optimization strategies include:

- Improving memory access patterns for coalescing
- Using shared memory for frequently accessed data
- Optimizing data layout

## Advanced PC Sampling Techniques

### Adjusting the Sampling Period

The sampling period affects the granularity and overhead of PC sampling:

```cpp
// For more detailed sampling (higher overhead)
config.samplingPeriod = 2;  // Sample every 2 cycles

// For less overhead (less detail)
config.samplingPeriod = 10; // Sample every 10 cycles
```

### Sampling Specific Kernels

To sample only specific kernels, you can enable and disable PC sampling around kernel launches:

```cpp
// Enable sampling before the kernel of interest
CUPTI_CALL(cuptiPCSamplingEnable(&config));

// Launch the kernel
myKernel<<<grid, block>>>(args);

// Disable sampling after the kernel
CUPTI_CALL(cuptiPCSamplingDisable());
```

### Analyzing Stall Reasons

PC sampling can provide information about why threads are stalled:

```cpp
// Check different stall reasons
const char* stallReasonNames[] = {
    "None", "Memory Throttle", "Not Selected", "Execution Dependency",
    "Synchronization", "Instruction Fetch", "Other"
};

for (int stall = 0; stall < CUPTI_PC_SAMPLING_STALL_INVALID; stall++) {
    if (entry->stallCount[stall] > 0) {
        printf("     %s stalls: %d\n", 
               stallReasonNames[stall], entry->stallCount[stall]);
    }
}
```

## Next Steps

- Apply PC sampling to your own CUDA kernels
- Focus optimization efforts on the identified hotspots
- Use PC sampling in conjunction with other profiling techniques
- Experiment with different sampling periods to balance detail and overhead 