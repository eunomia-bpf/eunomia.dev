# CUPTI Extensions

This directory contains additional utilities and extensions for the CUPTI samples.

## Overview

The extensions provide helper functions and utilities that simplify working with CUPTI APIs, particularly for more complex tasks like metric collection, evaluation, and result processing.

## Directory Structure

The extensions are organized into two main areas:

- **include**: Header files that define interfaces and utilities
  - **c_util**: Basic C utility functions (file operations, scope management, etc.)
  - **profilerhost_util**: Utilities for working with the CUPTI Profiler API

- **src**: Implementation of the utilities
  - **profilerhost_util**: Source code for the profiler host utilities

## Key Components

### Profiler Host Utilities

The profilerhost_util library provides functions for:

1. **Metric Management**:
   - Listing available metrics
   - Getting metric descriptions and properties
   - Converting between different metric formats

2. **Evaluation**:
   - Processing counter data
   - Calculating metric values from raw counter data
   - Interpreting profiling results

3. **File Operations**:
   - Reading/writing metric data
   - Managing profiling configurations

## Compatibility Notes

These extensions are designed to work with specific CUDA and CUPTI versions. Compatibility issues may arise when using with different versions of CUDA.

If you encounter build errors when building the profilerhost_util library:

1. Check your CUDA version compatibility
2. Use the dummy library created by the install.sh script for basic functionality
3. For full functionality, you may need to update the code to match your CUDA/CUPTI version

## Usage

Samples that require these extensions include:
- autorange_profiling
- userrange_profiling

These samples demonstrate more advanced CUPTI functionality that relies on the helper utilities provided in this directory.

## Building

The extensions are built automatically by the install.sh script in the main directory. However, if you need to build them manually:

```bash
cd src/profilerhost_util
make
cp libprofilerHostUtil.* ../../../lib64/
```

## See Also

- [CUPTI Profiler API Documentation](https://docs.nvidia.com/cuda/cupti/modules.html#group__CUPTI__PROFILER__API)
- [NVPERF API Documentation](https://docs.nvidia.com/cupti/Cupti/modules.html#group__NVPERF__API) 