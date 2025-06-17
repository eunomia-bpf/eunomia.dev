# CUPTI Program Counter (PC) Sampling Continuous Tutorial

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

## Helper Script Options

The `libpc_sampling_continuous.pl` script provides various configuration options:

```bash
# Show help
./libpc_sampling_continuous.pl --help

# Basic usage
./libpc_sampling_continuous.pl --app ./my_cuda_app

# Specify sampling frequency
./libpc_sampling_continuous.pl --app ./my_cuda_app --frequency 1000

# Set output file
./libpc_sampling_continuous.pl --app ./my_cuda_app --output samples.data

# Enable verbose output
./libpc_sampling_continuous.pl --app ./my_cuda_app --verbose
```

## Understanding the Output

### Sample Data Format

The PC sampling generates data files containing:

1. **Function Information**: Kernel names and their addresses
2. **PC Samples**: Program counter values with timestamps
3. **Stall Reasons**: Why warps were stalled at each sample point
4. **Source Correlation**: Assembly to source code mapping (when debug info is available)

### Example Output Structure

```
Kernel: vectorAdd(float*, float*, float*, int)
PC: 0x7f8b2c001000, Stall: MEMORY_DEPENDENCY, Count: 15
PC: 0x7f8b2c001008, Stall: EXECUTION_DEPENDENCY, Count: 8
PC: 0x7f8b2c001010, Stall: NOT_SELECTED, Count: 12
...
```

## Key Features

### Automatic Injection

The library automatically:
- Intercepts CUDA runtime and driver API calls
- Sets up PC sampling for each kernel launch
- Collects samples during kernel execution
- Generates detailed reports

### No Source Modification Required

Benefits include:
- Profile existing applications without recompilation
- Analyze third-party CUDA libraries
- Monitor production workloads
- Compare different optimization strategies

### Cross-Platform Support

The implementation handles:
- Different dynamic loading mechanisms (Linux vs Windows)
- Platform-specific library formats
- Varying CUDA installation paths
- Different debugging symbol formats

## Practical Applications

### Performance Hotspot Identification

Use PC sampling to:
1. **Find bottleneck instructions**: Identify assembly instructions where execution time is spent
2. **Analyze stall patterns**: Understand why warps are not making progress
3. **Optimize memory access**: Detect memory-bound operations
4. **Improve instruction scheduling**: Identify dependency stalls

### Algorithm Analysis

Apply to:
1. **Compare implementations**: Profile different algorithmic approaches
2. **Validate optimizations**: Measure the impact of code changes
3. **Understand GPU utilization**: See how well your code uses available resources
4. **Debug performance regressions**: Identify when and where performance degrades

## Advanced Usage

### Custom Sampling Configurations

Modify sampling behavior by:
1. **Adjusting sampling frequency**: Balance detail vs overhead
2. **Filtering specific kernels**: Focus on particular functions
3. **Setting duration limits**: Control profiling duration
4. **Configuring output formats**: Choose appropriate data formats

### Integration with Other Tools

Combine with:
1. **NVIDIA Nsight Compute**: Correlate with detailed metrics
2. **NVIDIA Nsight Systems**: Add timeline context
3. **Custom analysis scripts**: Process sampling data programmatically
4. **Visualization tools**: Create performance charts and graphs

## Troubleshooting

### Common Issues

1. **Library not found**: Ensure all paths are correctly set in LD_LIBRARY_PATH/PATH
2. **Permission errors**: Check that the target application has necessary permissions
3. **CUDA version mismatch**: Verify CUPTI version matches CUDA runtime
4. **Missing symbols**: Ensure debug information is available for source correlation

### Debug Tips

1. **Enable verbose logging**: Use `--verbose` flag to see detailed execution
2. **Check dependencies**: Verify all required libraries are accessible
3. **Test with simple apps**: Start with basic CUDA samples before complex applications
4. **Monitor resource usage**: Ensure sufficient memory for sampling data

## Next Steps

- Experiment with different sampling frequencies to find optimal settings
- Apply continuous PC sampling to your own CUDA applications
- Combine with the `pc_sampling_utility` to analyze collected data
- Explore correlation with source code using debug symbols
- Integrate PC sampling data with your performance analysis workflow 