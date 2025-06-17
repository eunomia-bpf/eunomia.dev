# CUPTI Trace Injection Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

The CUPTI Trace Injection sample demonstrates how to create a lightweight tracing library that can be automatically injected into any CUDA application. This approach enables comprehensive activity tracing without requiring source code modifications, making it perfect for profiling existing applications, third-party libraries, or production workloads.

## What You'll Learn

- How to build an injection library for automatic CUDA activity tracing
- Understanding the CUDA injection mechanism for seamless integration
- Implementing NVTX activity recording for enhanced timeline visualization
- Cross-platform injection techniques (Linux and Windows)
- Collecting comprehensive trace data without application modifications

## Understanding Trace Injection

Trace injection provides several key advantages for CUDA profiling:

1. **Zero application modification**: Profile any CUDA application without recompilation
2. **Automatic activation**: CUDA runtime loads and initializes the tracing automatically
3. **Comprehensive coverage**: Captures all CUDA operations and activities
4. **NVTX integration**: Records user-defined ranges and markers
5. **Timeline visualization**: Generates data suitable for timeline analysis tools

## Architecture Overview

The trace injection system consists of:

1. **Injection Library**: `libcupti_trace_injection.so` (Linux) or `libcupti_trace_injection.dll` (Windows)
2. **CUDA Injection Hook**: Automatic loading via `CUDA_INJECTION64_PATH`
3. **NVTX Integration**: Optional NVTX activity recording via `NVTX_INJECTION64_PATH`
4. **Activity Collection**: Comprehensive CUDA API and GPU activity tracing
5. **Output Generation**: Structured trace data for analysis tools

## Key Features

### Automatic Initialization
- No source code changes required
- Works with any CUDA application
- Supports both runtime and driver APIs
- Handles complex multi-threaded applications

### Comprehensive Activity Tracing
- CUDA runtime API calls
- CUDA driver API calls
- Kernel execution activities
- Memory transfer operations
- Context and stream management

### NVTX Support
- User-defined range recording
- Custom markers and annotations
- Enhanced timeline visualization
- Application phase correlation

## Building the Sample

### Prerequisites

Ensure you have:
- CUDA Toolkit with CUPTI
- Development tools (gcc/Visual Studio)
- For Windows: Microsoft Detours library

### Linux Build Process

1. Navigate to the sample directory:
   ```bash
   cd cupti_trace_injection
   ```

2. Build using the provided Makefile:
   ```bash
   make
   ```
   
   This creates `libcupti_trace_injection.so`.

### Windows Build Process

For Windows builds, you need the Microsoft Detours library:

1. **Download and build Detours**:
   ```cmd
   # Download from https://github.com/microsoft/Detours
   # Extract to a folder
   cd Detours
   set DETOURS_TARGET_PROCESSOR=X64
   "Program Files (x86)\Microsoft Visual Studio\<version>\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
   NMAKE
   ```

2. **Copy required files**:
   ```cmd
   copy detours.h <cupti_trace_injection_folder>
   copy detours.lib <cupti_trace_injection_folder>
   ```

3. **Build the sample**:
   ```cmd
   nmake
   ```
   
   This creates `libcupti_trace_injection.dll`.

## Running the Sample

### Linux Usage

1. **Set up injection environment**:
   ```bash
   export CUDA_INJECTION64_PATH=/full/path/to/libcupti_trace_injection.so
   export NVTX_INJECTION64_PATH=/full/path/to/libcupti.so
   ```

2. **Run your CUDA application**:
   ```bash
   ./your_cuda_application
   ```

### Windows Usage

1. **Set up injection environment**:
   ```cmd
   set CUDA_INJECTION64_PATH=C:\full\path\to\libcupti_trace_injection.dll
   set NVTX_INJECTION64_PATH=C:\full\path\to\cupti.dll
   ```

2. **Run your CUDA application**:
   ```cmd
   your_cuda_application.exe
   ```

### Environment Variables

#### CUDA_INJECTION64_PATH
Specifies the path to your injection library. When set, CUDA automatically:
- Loads the shared library at initialization
- Calls the `InitializeInjection()` function
- Enables tracing for all subsequent CUDA operations

#### NVTX_INJECTION64_PATH
Optional path to CUPTI library for NVTX activity recording:
- Enables user-defined range collection
- Records custom markers and annotations
- Provides enhanced timeline context

## Understanding the Output

### Trace Data Format

The injection library generates comprehensive trace data including:

```
CUDA Runtime API Calls:
  cudaMalloc: Start=1234567890, End=1234567925, Duration=35μs
  cudaMemcpy: Start=1234567950, End=1234568100, Duration=150μs
  cudaLaunchKernel: Start=1234568150, End=1234568175, Duration=25μs

GPU Activities:
  Kernel: vectorAdd, Start=1234568200, End=1234568500, Duration=300μs
  MemcpyHtoD: Size=4096KB, Start=1234567950, End=1234568100, Duration=150μs
  MemcpyDtoH: Size=4096KB, Start=1234568600, End=1234568750, Duration=150μs

NVTX Ranges:
  Range: "Data Preparation", Start=1234567800, End=1234568150, Duration=350μs
  Range: "Computation", Start=1234568150, End=1234568550, Duration=400μs
  Range: "Result Validation", Start=1234568600, End=1234568900, Duration=300μs
```

### Key Metrics

1. **API Call Timing**: Duration of CUDA runtime and driver API calls
2. **GPU Activity Timeline**: Actual kernel execution and memory transfer times
3. **Memory Usage**: Allocation sizes and transfer patterns
4. **Concurrency Analysis**: Overlapping operations and stream utilization
5. **User-Defined Context**: NVTX ranges providing application semantics

## Practical Applications

### Performance Analysis

Use trace injection for:
- **Bottleneck identification**: Find the slowest operations in your application
- **Concurrency analysis**: Understand how well operations overlap
- **Memory bandwidth utilization**: Analyze data transfer efficiency
- **API overhead measurement**: Quantify CUDA API call costs

### Timeline Visualization

Trace data can be imported into:
- **NVIDIA Nsight Systems**: Comprehensive timeline analysis
- **Chrome Tracing**: Web-based visualization
- **Custom analysis tools**: Programmatic trace processing
- **Performance comparison tools**: Before/after optimization analysis

### Production Monitoring

Deploy in production environments to:
- Monitor application performance over time
- Detect performance regressions
- Analyze real-world workload patterns
- Generate automated performance reports

## Advanced Usage

### Custom Activity Filtering

Modify the injection library to focus on specific activities:

```cpp
// Filter specific API calls
bool shouldTraceAPI(const char* apiName) {
    return (strstr(apiName, "Launch") != nullptr ||
            strstr(apiName, "Memcpy") != nullptr);
}

// Filter kernel activities
bool shouldTraceKernel(const char* kernelName) {
    return !strstr(kernelName, "internal_");
}
```

### Enhanced NVTX Integration

Leverage NVTX for better application context:

```cpp
// In your application (optional, but enhances tracing)
nvtxRangePush("Critical Section");
// ... CUDA operations ...
nvtxRangePop();

nvtxMark("Checkpoint A");
```

### Multi-GPU Analysis

The injection library automatically handles:
- Multiple GPU contexts
- Cross-device memory transfers
- Peer-to-peer communications
- Device-specific activity timelines

## Output Formats and Analysis

### Raw Data Processing

```bash
# Convert trace data to various formats
./process_trace_data --input trace.cupti --output timeline.json --format chrome

# Generate performance summary
./analyze_trace --input trace.cupti --summary performance_report.txt

# Compare multiple traces
./compare_traces --baseline baseline.cupti --optimized optimized.cupti
```

### Integration with Analysis Tools

```python
# Python analysis example
import cupti_trace_parser

trace = cupti_trace_parser.load('trace.cupti')
kernel_times = trace.get_kernel_durations()
api_overhead = trace.get_api_overhead()

print(f"Total kernel time: {sum(kernel_times)}μs")
print(f"Average API overhead: {api_overhead.mean()}μs")
```

## Troubleshooting

### Common Issues

1. **Library not loaded**: Verify the full path in environment variables
2. **Permission errors**: Ensure proper file and directory permissions
3. **Missing dependencies**: Check that all required libraries are available
4. **NVTX not working**: Verify NVTX_INJECTION64_PATH points to correct CUPTI library

### Debug Tips

1. **Test with simple applications**: Start with basic CUDA samples
2. **Check environment setup**: Verify all paths are correct and accessible
3. **Enable verbose logging**: Add debug output to the injection library
4. **Monitor library loading**: Use system tools to verify injection is working

### Platform-Specific Notes

**Linux**:
- Use `ldd` to check library dependencies
- Verify `LD_LIBRARY_PATH` includes required directories
- Check that shared libraries have execute permissions

**Windows**:
- Use Dependency Walker to analyze DLL dependencies
- Ensure all DLLs are in the system PATH or application directory
- Verify that Visual C++ redistributables are installed

## Next Steps

- Apply trace injection to profile your own CUDA applications
- Experiment with different NVTX annotations to enhance trace context
- Develop custom analysis scripts for your specific performance metrics
- Integrate trace collection into your development and deployment workflows
- Combine with other CUPTI samples for comprehensive performance analysis 