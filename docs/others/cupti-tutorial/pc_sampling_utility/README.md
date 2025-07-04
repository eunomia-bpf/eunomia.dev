# CUPTI PC Sampling Analysis Utility Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

The PC Sampling Utility is a powerful post-processing tool that analyzes data collected by the `pc_sampling_continuous` sample. It transforms raw PC sampling data into actionable insights by correlating assembly instructions with stall reasons and providing source-level mapping when debug information is available.

## What You'll Learn

- How to analyze PC sampling data files generated by continuous sampling
- Understanding stall reason counters at the assembly instruction level
- Techniques for correlating assembly code with CUDA C source code
- Working with CUDA cubin files for detailed analysis
- Interpreting performance bottlenecks from PC sampling results

## Understanding PC Sampling Data Analysis

PC sampling analysis differs from real-time monitoring because it:

1. **Processes collected data offline**: Allows detailed analysis without runtime overhead
2. **Provides assembly-level insights**: Shows exactly which instructions cause performance issues
3. **Correlates with source code**: Maps performance hotspots back to your original C/C++ code
4. **Quantifies stall reasons**: Explains why GPU execution units are idle
5. **Supports batch processing**: Can analyze multiple sampling sessions together

## Key Concepts

### Stall Reasons

GPU warps can be stalled for various reasons:

- **MEMORY_DEPENDENCY**: Waiting for memory operations to complete
- **EXECUTION_DEPENDENCY**: Waiting for previous instructions in the pipeline
- **NOT_SELECTED**: Warp is ready but scheduler chose other warps
- **MEMORY_THROTTLE**: Memory subsystem is saturated
- **PIPE_BUSY**: Execution pipeline is fully utilized
- **CONSTANT_MEMORY_DEPENDENCY**: Waiting for constant memory access
- **TEXTURE_MEMORY_DEPENDENCY**: Waiting for texture memory access

### Assembly to Source Correlation

The utility can map assembly instructions back to source code when:
- Debug information is compiled into the application (`-g` flag)
- CUDA cubin files are extracted and properly named
- Source files are accessible at analysis time

## Building the Utility

### Prerequisites

Ensure you have:
- CUDA Toolkit installed
- CUPTI libraries available
- Access to cubin files from your target application

### Build Process

1. Navigate to the pc_sampling_utility directory:
   ```bash
   cd pc_sampling_utility
   ```

2. Build using the provided Makefile:
   ```bash
   make
   ```
   
   This creates the `pc_sampling_utility` executable.

## Preparing Input Data

### Generating PC Sampling Data

First, collect PC sampling data using the continuous sampling library:

```bash
# Using the continuous sampling library
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cupti/lib64:/path/to/pc_sampling_continuous
./libpc_sampling_continuous.pl --app ./your_cuda_application --output samples.data
```

### Extracting CUDA Cubin Files

For source correlation, extract cubin files from your application:

```bash
# Extract all cubin files from executable
cuobjdump -xelf all your_cuda_application

# Extract from library files
cuobjdump -xelf all libmy_cuda_library.so
```

**Important**: The `cuobjdump` version must match the CUDA Toolkit version used to build your application.

### Naming Cubin Files

Rename the extracted cubin files sequentially:

```bash
# Rename cubin files in order
mv first_extracted_file.cubin 1.cubin
mv second_extracted_file.cubin 2.cubin
mv third_extracted_file.cubin 3.cubin
# ... and so on
```

The utility expects cubin files to be named `1.cubin`, `2.cubin`, `3.cubin`, etc.

## Running the Analysis

### Basic Usage

```bash
./pc_sampling_utility --input samples.data
```

### Command Line Options

View all available options:

```bash
./pc_sampling_utility --help
```

Common options include:

```bash
# Specify input file
./pc_sampling_utility --input samples.data

# Set cubin directory
./pc_sampling_utility --input samples.data --cubin-path ./cubins/

# Enable verbose output
./pc_sampling_utility --input samples.data --verbose

# Filter specific kernels
./pc_sampling_utility --input samples.data --kernel vectorAdd

# Set output format
./pc_sampling_utility --input samples.data --format csv
```

## Understanding the Output

### Sample Output Format

```
Kernel: vectorAdd(float*, float*, float*, int)
================================================================================

Assembly Analysis:
PC: 0x008 | INST: LDG.E.SYS R2, [R8] | Stall: MEMORY_DEPENDENCY | Count: 245 (15.3%)
PC: 0x010 | INST: LDG.E.SYS R4, [R10] | Stall: MEMORY_DEPENDENCY | Count: 198 (12.4%)
PC: 0x018 | INST: FADD R6, R2, R4 | Stall: EXECUTION_DEPENDENCY | Count: 89 (5.6%)
PC: 0x020 | INST: STG.E.SYS [R12], R6 | Stall: MEMORY_DEPENDENCY | Count: 156 (9.7%)

Source Correlation:
PC: 0x008 | File: vector_add.cu | Line: 42 | Code: float a = A[i];
PC: 0x010 | File: vector_add.cu | Line: 43 | Code: float b = B[i];
PC: 0x018 | File: vector_add.cu | Line: 44 | Code: float result = a + b;
PC: 0x020 | File: vector_add.cu | Line: 45 | Code: C[i] = result;

Performance Summary:
Total Samples: 1599
Memory Bound: 599 samples (37.5%)
Execution Bound: 234 samples (14.6%)
Scheduler Limited: 445 samples (27.8%)
Other: 321 samples (20.1%)
```

### Key Metrics to Analyze

1. **Stall Distribution**: Which stall reasons dominate your kernel execution
2. **Hotspot Instructions**: Assembly instructions with the highest sample counts
3. **Memory Access Patterns**: How memory operations contribute to stalls
4. **Source Line Correlation**: Which source lines correspond to performance issues

## Practical Analysis Workflows

### Identifying Memory Bottlenecks

1. **Look for MEMORY_DEPENDENCY stalls**: High counts indicate memory-bound kernels
2. **Analyze access patterns**: Check if accesses are coalesced
3. **Consider caching strategies**: Evaluate shared memory or texture memory usage

Example workflow:
```bash
# Focus on memory-related stalls
./pc_sampling_utility --input samples.data --filter-stall MEMORY_DEPENDENCY

# Analyze specific memory instructions
./pc_sampling_utility --input samples.data --filter-instruction "LDG\|STG"
```

### Optimizing Instruction Dependencies

1. **Identify EXECUTION_DEPENDENCY hotspots**: Shows instruction pipeline stalls
2. **Analyze instruction ordering**: Look for opportunities to reorder operations
3. **Consider ILP (Instruction Level Parallelism)**: Find independent operations

### Understanding Scheduler Behavior

1. **Monitor NOT_SELECTED stalls**: Indicates scheduler pressure
2. **Analyze warp utilization**: Check if enough warps are available
3. **Consider occupancy optimization**: Increase warps per SM when possible

## Advanced Analysis Techniques

### Comparing Multiple Runs

```bash
# Analyze baseline version
./pc_sampling_utility --input baseline.data --output baseline_analysis.txt

# Analyze optimized version
./pc_sampling_utility --input optimized.data --output optimized_analysis.txt

# Compare results
diff baseline_analysis.txt optimized_analysis.txt
```

### Statistical Analysis

```bash
# Generate CSV output for spreadsheet analysis
./pc_sampling_utility --input samples.data --format csv --output analysis.csv

# Create histograms of stall reasons
./pc_sampling_utility --input samples.data --histogram --bins 20
```

### Kernel-Specific Analysis

```bash
# Analyze only specific kernels
./pc_sampling_utility --input samples.data --kernel "matrixMul.*"

# Exclude certain kernels
./pc_sampling_utility --input samples.data --exclude-kernel "memcpy.*"
```

## Integration with Development Workflow

### Performance Regression Detection

1. **Baseline establishment**: Create performance profiles for known-good versions
2. **Automated analysis**: Include PC sampling in CI/CD pipelines
3. **Threshold monitoring**: Alert on significant performance changes

### Optimization Guidance

1. **Hotspot identification**: Focus optimization efforts on high-impact areas
2. **Validation**: Verify that optimizations reduce relevant stall reasons
3. **Iteration**: Use sampling data to guide successive optimization attempts

## Troubleshooting

### Common Issues

1. **Missing cubin files**: Ensure cubins are extracted and properly named
2. **Version mismatches**: Verify cuobjdump version matches CUDA Toolkit
3. **Missing debug info**: Compile with `-g` flag for source correlation
4. **Path issues**: Check that cubin files are in the expected location

### Debug Tips

1. **Start with simple kernels**: Test the workflow with basic examples first
2. **Verify cubin extraction**: Check that cuobjdump produces valid files
3. **Test without source correlation**: Ensure basic assembly analysis works
4. **Use verbose output**: Enable detailed logging to understand processing steps

## Next Steps

- Apply PC sampling analysis to identify performance bottlenecks in your applications
- Integrate the analysis workflow into your optimization process
- Experiment with different sampling configurations to balance detail and overhead
- Combine PC sampling results with other profiling tools for comprehensive analysis
- Develop custom scripts to automate analysis for your specific use cases 