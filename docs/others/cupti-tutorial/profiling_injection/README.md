# CUPTI Profiling API Injection Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

The CUPTI Profiling API Injection sample demonstrates how to create a profiling library that can be injected into any CUDA application without requiring source code modifications. This powerful technique allows you to collect detailed performance metrics from existing applications using CUDA's injection mechanism.

## What You'll Learn

- How to build a shared library for CUPTI profiling injection
- Understanding the CUDA injection mechanism (`CUDA_INJECTION64_PATH`)
- Implementing callback-based profiling for kernel launches and context creation
- Using the Profiler API with Kernel Replay and Auto Range modes
- Collecting and analyzing GPU performance metrics in real-time

## Understanding Profiling Injection

Profiling injection offers several advantages over traditional profiling approaches:

1. **No source code modification**: Profile any CUDA application without recompilation
2. **Automatic initialization**: CUDA loads and initializes your profiling library automatically
3. **Comprehensive coverage**: Intercepts all CUDA operations in the target application
4. **Flexible configuration**: Control profiling behavior through environment variables
5. **Production-ready**: Can be used with release builds and third-party applications

## Architecture Overview

The injection system consists of several key components:

1. **Injection Library**: `libinjection.so` - The main profiling library
2. **CUDA Injection Mechanism**: Automatic loading via `CUDA_INJECTION64_PATH`
3. **Callback System**: Intercepts CUDA API calls for profiling setup
4. **Profiler API Integration**: Configures metric collection for each context
5. **Target Applications**: `simple_target` and `complex_target` for testing

## Sample Applications

### simple_target
A basic executable that calls a kernel several times with increasing work per call. Perfect for:
- Testing the injection mechanism
- Understanding basic profiling workflows
- Validating metric collection

### complex_target
A sophisticated example featuring multiple kernel launch patterns:
- Default stream execution
- Multiple stream concurrency
- Multi-device execution (when available)
- Thread-based parallelism

This mirrors the `concurrent_profiling` sample complexity and demonstrates that injection handles diverse execution patterns.

## Building the Sample

### Prerequisites

Ensure you have:
- CUDA Toolkit with CUPTI
- profilerHostUtils library (built from `cuda/extras/CUPTI/samples/extensions/src/profilerhost_util/`)
- Appropriate development tools (gcc, make)

### Build Process

```bash
# Set CUDA installation path
export CUDA_INSTALL_PATH=/path/to/cuda

# Build all components
make CUDA_INSTALL_PATH=/path/to/cuda
```

This creates three build targets:
1. `libinjection.so` - The injection library
2. `simple_target` - Basic test application
3. `complex_target` - Advanced test application

### Build Components

#### libinjection.so
The core profiling library that:
- Registers callbacks for `cuLaunchKernel` and context creation
- Creates Profiler API configurations for each context
- Configures Kernel Replay and Auto Range modes
- Tracks kernel launches and manages profiling passes
- Prints metrics when passes complete or at exit

#### Target Applications
Both target applications provide different complexity levels for testing:
- `simple_target`: Sequential kernel launches with varying work
- `complex_target`: Concurrent execution patterns across multiple streams and devices

## Configuration Options

### Environment Variables

#### INJECTION_KERNEL_COUNT
Controls how many kernels to include in a single profiling session:

```bash
export INJECTION_KERNEL_COUNT=20  # Default is 10
```

When this many kernels have launched, the session ends and metrics are printed, then a new session begins.

#### INJECTION_METRICS
Specifies which metrics to collect:

```bash
# Default metrics
export INJECTION_METRICS="sm__cycles_elapsed.avg smsp__sass_thread_inst_executed_op_dadd_pred_on.avg smsp__sass_thread_inst_executed_op_dfma_pred_on.avg"

# Custom metrics (space, comma, or semicolon separated)
export INJECTION_METRICS="sm__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum"
```

Default metrics focus on:
- `sm__cycles_elapsed.avg`: Overall execution time
- `smsp__sass_thread_inst_executed_op_dadd_pred_on.avg`: Double-precision add operations
- `smsp__sass_thread_inst_executed_op_dfma_pred_on.avg`: Double-precision fused multiply-add operations

## Running the Sample

### Basic Usage

Set the injection path and run your target application:

```bash
env CUDA_INJECTION64_PATH=./libinjection.so ./simple_target
```

### Advanced Configuration

```bash
# Configure kernel count and custom metrics
env CUDA_INJECTION64_PATH=./libinjection.so \
    INJECTION_KERNEL_COUNT=15 \
    INJECTION_METRICS="sm__cycles_elapsed.avg,dram__bytes_read.sum" \
    ./complex_target
```

### Testing Different Patterns

```bash
# Test with simple target
env CUDA_INJECTION64_PATH=./libinjection.so ./simple_target

# Test with complex multi-stream target
env CUDA_INJECTION64_PATH=./libinjection.so ./complex_target

# Test with your own application
env CUDA_INJECTION64_PATH=./libinjection.so /path/to/your/cuda/app
```

## Understanding the Output

### Sample Output Format

```
=== Profiling Session 1 ===
Context: 0x7f8b2c000000
Kernels in session: 10

Metric Results:
sm__cycles_elapsed.avg: 125434.2
smsp__sass_thread_inst_executed_op_dadd_pred_on.avg: 8192.0
smsp__sass_thread_inst_executed_op_dfma_pred_on.avg: 16384.0

=== Profiling Session 2 ===
Context: 0x7f8b2c000000
Kernels in session: 10
...
```

### Key Information

1. **Session Boundaries**: Each session contains a configurable number of kernels
2. **Context Information**: Shows which CUDA context the metrics apply to
3. **Metric Values**: Collected performance data for the specified metrics
4. **Automatic Rotation**: New sessions start automatically when kernel count is reached

## Code Architecture

### Injection Entry Point

```cpp
extern "C" void InitializeInjection(void)
{
    // Called automatically when CUDA loads the injection library
    // Register callbacks and set up profiling infrastructure
}
```

### Callback Registration

The library registers callbacks for:

1. **Context Creation**: Sets up Profiler API configuration for new contexts
2. **Kernel Launches**: Tracks launch count and manages session boundaries
3. **cuLaunchKernel**: Handles most kernel launch scenarios
4. **Additional Launches**: May need `cuLaunchCooperativeKernel` or `cuLaunchGrid` for some applications

### Profiler API Integration

For each context, the library:
1. Creates a Profiler API configuration
2. Enables Kernel Replay mode for detailed metrics
3. Sets up Auto Range mode for automatic kernel grouping
4. Configures the specified metrics for collection

## Advanced Usage Scenarios

### Multi-GPU Applications

The injection library automatically handles multi-GPU scenarios by:
- Creating separate profiling configurations for each device context
- Tracking kernel launches per context independently
- Reporting metrics separately for each GPU

### Multi-Threaded Applications

Thread safety is handled through:
- Context-specific profiling configurations
- Thread-local callback handling
- Synchronized metric collection and reporting

### Long-Running Applications

For applications with many kernels:
- Automatic session rotation prevents memory overflow
- Configurable kernel count per session
- Real-time metric reporting during execution

## Extending the Sample

### Adding New Callbacks

To handle additional kernel launch types:

```cpp
// Register additional callbacks in InitializeInjection
CUPTI_CALL(cuptiSubscribe(&subscriber, 
           (CUpti_CallbackFunc)callbackHandler, 
           &callbackData));

CUPTI_CALL(cuptiEnableCallback(1, subscriber, 
           CUPTI_CB_DOMAIN_DRIVER_API, 
           CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel));
```

### Custom Metrics

Add application-specific metrics:

```cpp
// Define custom metric sets based on your analysis needs
const char* memoryMetrics[] = {
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
};
```

### Output Formatting

Customize output format for integration with analysis tools:

```cpp
// Add JSON, CSV, or other structured output formats
void printMetricsJSON(const std::vector<MetricResult>& results);
void printMetricsCSV(const std::vector<MetricResult>& results);
```

## Integration with Development Workflow

### Continuous Integration

Integrate injection profiling into CI/CD:

```bash
# Automated performance testing
env CUDA_INJECTION64_PATH=./libinjection.so \
    INJECTION_METRICS="sm__cycles_elapsed.avg" \
    ./run_performance_tests.sh
```

### Performance Monitoring

Use for production monitoring:
- Deploy injection library with production applications
- Collect performance baselines
- Monitor for performance regressions
- Generate automated performance reports

## Troubleshooting

### Common Issues

1. **Library not loaded**: Verify `CUDA_INJECTION64_PATH` points to correct library
2. **Missing symbols**: Ensure profilerHostUtils is properly linked
3. **Callback not triggered**: Check that target application uses supported launch APIs
4. **Metric collection fails**: Verify metrics are available on target GPU

### Debug Tips

1. **Enable verbose logging**: Add debug output to callback functions
2. **Test with simple applications**: Start with basic CUDA samples
3. **Check CUPTI version**: Ensure compatibility with CUDA runtime version
4. **Verify library dependencies**: Use `ldd` to check shared library dependencies

## Next Steps

- Experiment with different metric combinations to understand GPU behavior
- Apply injection profiling to your own CUDA applications
- Extend the sample to collect additional performance data
- Integrate with visualization tools for performance analysis
- Develop custom analysis workflows based on collected metrics 