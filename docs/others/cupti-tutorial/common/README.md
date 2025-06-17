# CUPTI Common Helper Files

## Overview

This directory contains shared helper files and utilities used across multiple CUPTI samples. These files provide common functionality for CUPTI initialization, error handling, activity record processing, and other frequently used operations.

## Files

### helper_cupti.h

A comprehensive header file providing:

- **Error Handling Macros**: Simplified error checking for CUPTI and CUDA API calls
- **Common Initialization Functions**: Standard setup routines for CUPTI profiling
- **Memory Management Utilities**: Safe allocation and deallocation helpers
- **Device Management Functions**: GPU device enumeration and selection utilities

Key macros and functions include:

```cpp
// Error checking macros
#define CUPTI_API_CALL(apiFuncCall)
#define RUNTIME_API_CALL(apiFuncCall)  
#define DRIVER_API_CALL(apiFuncCall)
#define MEMORY_ALLOCATION_CALL(var)

// Common initialization
void initCuda();
void cleanupCuda();
CUdevice pickDevice();
```

### helper_cupti_activity.h

An extensive header file for CUPTI activity record processing:

- **Activity Record Management**: Structures and functions for handling different activity types
- **Buffer Management**: Efficient buffer allocation and processing for activity records
- **Callback Registration**: Utilities for setting up CUPTI callbacks
- **Data Extraction**: Helper functions to extract meaningful data from activity records
- **Output Formatting**: Functions for pretty-printing profiling results

Key features include:

```cpp
// Activity record processing
void processActivityRecord(CUpti_Activity* record);
void handleActivityBuffer(uint8_t* buffer, size_t validSize);

// Buffer management
CUpti_BufferAlloc bufferRequested;
CUpti_BufferCompleted bufferCompleted;

// Callback utilities
void enableActivityRecords();
void registerCallbacks();
```

## Usage in Samples

These helper files are included in most CUPTI samples to:

1. **Reduce Code Duplication**: Common operations are centralized
2. **Simplify Error Handling**: Consistent error checking across samples
3. **Standardize Initialization**: Uniform CUPTI setup procedures
4. **Streamline Activity Processing**: Reusable activity record handling

## Typical Include Pattern

```cpp
#include "helper_cupti.h"
#include "helper_cupti_activity.h"

int main() {
    // Initialize CUDA and CUPTI
    initCuda();
    
    // Setup profiling
    enableActivityRecords();
    registerCallbacks();
    
    // Run your application code
    runKernels();
    
    // Process results and cleanup
    processResults();
    cleanupCuda();
    
    return 0;
}
```

## Integration with Samples

Most samples in this repository use these helpers by:

1. Including the appropriate header files
2. Calling initialization functions in main()
3. Using error checking macros throughout
4. Leveraging activity processing utilities
5. Following standard cleanup procedures

## Customization

While these helpers provide common functionality, individual samples may:

- Extend the provided structures
- Add sample-specific processing functions
- Customize callback handlers
- Implement additional error handling

## Benefits

Using these common helpers provides:

- **Consistency**: All samples follow similar patterns
- **Reliability**: Well-tested error handling and initialization
- **Maintainability**: Changes to common functionality affect all samples
- **Learning**: Clear examples of CUPTI best practices
- **Efficiency**: Optimized buffer management and processing

## Contributing

When adding new samples or modifying existing ones:

1. Use the existing helper functions when possible
2. Add new common functionality to the appropriate helper file
3. Maintain consistent error handling patterns
4. Update documentation when adding new helpers
5. Test changes across multiple samples to ensure compatibility

These helper files are essential for maintaining consistency and reliability across the CUPTI sample suite. 