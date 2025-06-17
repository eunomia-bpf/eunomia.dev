# NVIDIA CUPTI Tutorials

This repository contains tutorial applications demonstrating the use of the CUDA Profiling Tools Interface (CUPTI). CUPTI provides performance analysis tools with detailed information about how applications are using the GPU through a standardized interface. Each sample is structured as a comprehensive tutorial to help you understand and apply specific CUPTI features.

The samples are from NVIDIA's CUPTI samples.

## Overview

CUPTI enables the creation of profiling and tracing tools that target CUDA applications. These tutorials demonstrate how to use the following capabilities:
- Activity API for asynchronous collection of CUDA runtime and driver API routines and GPU activities
- Callback API for intercepting CUDA runtime and driver API calls
- Event API for accessing hardware performance counters
- Metric API for accessing computed metrics
- Profiling API for controlling the profiler from within the application

## Requirements

- CUDA Toolkit (compatible with the tutorials)
- GPU with compute capability 2.0 or higher
- Appropriate GPU drivers

## Installation

To set up the required dependencies and environment for building the tutorials, run the included installation script:

```bash
chmod +x install.sh  # Make the script executable (if not already)
./install.sh         # Run the installation script
source setup_env.sh  # Set up the environment variables
```

The install script will:
1. Check for a valid CUDA installation
2. Create a local lib64 directory with necessary symlinks
3. Build required extension libraries
4. Set up environment variables for running the tutorials

## Building the Tutorials

Each tutorial includes its own Makefile. You can build all tutorials at once using the provided top-level Makefile:

```bash
make
```

To build a specific tutorial:

```bash
cd <tutorial_directory>
make
```

## Tutorial Descriptions

Each tutorial provides a comprehensive walkthrough of a specific CUPTI feature, including explanations of concepts, step-by-step code analysis, execution instructions, and output interpretation.

### Activity API Tutorials
- **activity_trace**: Learn how to trace CUDA API calls and GPU activities, with detailed timeline collection and analysis
- **activity_trace_async**: Master asynchronous activity tracing for improved performance when profiling large applications
- **userrange_profiling**: Discover how to define custom ranges in your code and collect performance metrics within these specific sections
- **autorange_profiling**: Understand automatic range profiling for kernel-level performance analysis

### Callback API Tutorials
- **callback_event**: Learn how to intercept CUDA events and implement custom handlers
- **callback_timestamp**: Discover techniques for collecting precise GPU timestamps for CUDA operations
- **callback_metric**: Master the collection of GPU performance metrics during CUDA API calls

### Event and Metric API Tutorials
- **cupti_query**: Explore how to query available CUPTI domains, events, and metrics for your specific GPU
- **event_sampling**: Learn techniques for sampling hardware performance counters during kernel execution
- **event_multi_gpu**: Master event collection across multiple GPUs in a single system

### Advanced Feature Tutorials
- **nvlink_bandwidth**: Learn to detect NVLink connections and monitor data transfer rates between GPUs
- **openacc_trace**: Understand how to trace OpenACC API calls and correlate them with GPU activities
- **pc_sampling**: Master program counter sampling to identify hotspots in your CUDA kernels
- **sass_source_map**: Learn how to map SASS assembly instructions back to your source code
- **unified_memory**: Discover techniques for profiling and optimizing Unified Memory operations

## Tutorial Structure

Each tutorial directory contains:
- Source code files (.cpp, .cu) with detailed comments explaining the implementation
- Makefile for building the tutorial
- README.md with a comprehensive explanation of:
  - Concepts and theory behind the feature
  - Step-by-step code walkthrough
  - Instructions for running and experimenting
  - Analysis of the expected output
  - Advanced usage tips and optimization strategies

## Learning Path

For those new to CUPTI, we recommend following this learning path:
1. Start with **cupti_query** to understand available metrics and events
2. Move to **activity_trace** to learn basic tracing capabilities
3. Explore **callback_timestamp** and **callback_event** to understand API interception
4. Continue with more advanced tutorials based on your specific needs

## Troubleshooting

If you encounter build errors:
1. Ensure CUDA is properly installed and CUDA_HOME is set
2. Run the install.sh script to set up dependencies
3. Check that library paths are correctly set in setup_env.sh
4. For tutorials requiring additional libraries (like profilerHostUtil), make sure they are built correctly

## License

These tutorials are provided under the NVIDIA Corporation license. 