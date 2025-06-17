# CUPTI Checkpoint API Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

The CUPTI Checkpoint API provides a powerful mechanism for capturing and restoring GPU device state, enabling reproducible kernel execution even when kernels modify their own input data. This tutorial demonstrates how to use checkpoints to ensure consistent results across multiple kernel invocations.

## What You'll Learn

- How to use CUPTI's checkpoint API to save and restore GPU state
- Techniques for ensuring reproducible kernel execution
- Understanding when checkpoints are necessary for correctness
- Managing device memory state across kernel invocations
- Best practices for checkpoint-based debugging and testing

## Understanding the Problem

Many CUDA kernels modify their input data during execution, which can lead to different results when the same kernel is run multiple times. This is particularly common in:

- **Reduction operations** that overwrite input arrays
- **In-place transformations** that modify data during processing
- **Iterative algorithms** that use the same buffer for input and output
- **Debugging scenarios** where you want to replay the exact same conditions

## The Checkpoint Solution

CUPTI's checkpoint API allows you to:
1. **Save** the complete state of GPU memory at a specific point
2. **Restore** that exact state later, ensuring identical conditions
3. **Replay** kernel executions with guaranteed reproducibility

## Code Architecture

### Checkpoint Structure

```cpp
// Configure a checkpoint object
CUpti_Checkpoint cp = { CUpti_Checkpoint_STRUCT_SIZE };
cp.ctx = context;              // CUDA context to checkpoint
cp.optimizations = 1;          // Enable optimizations
```

### Basic Checkpoint Workflow

```cpp
// 1. Save checkpoint before first kernel execution
CUPTI_API_CALL(cuptiCheckpointSave(&cp));

// 2. Run kernel (may modify input data)
MyKernel<<<blocks, threads>>>(deviceData, size);

// 3. For subsequent runs, restore checkpoint first
CUPTI_API_CALL(cuptiCheckpointRestore(&cp));

// 4. Run kernel again with identical initial conditions
MyKernel<<<blocks, threads>>>(deviceData, size);
```

## Sample Walkthrough

### The Problem Kernel

Our sample uses a reduction kernel that demonstrates the issue:

```cpp
__global__ void Reduce(float *pData, size_t N)
{
    float totalSumData = 0.0;

    // Each thread sums its elements locally
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        totalSumData += pData[i];
    }

    // Save per-thread sum back to input array (MODIFIES INPUT!)
    pData[threadIdx.x] = totalSumData;
    
    __syncthreads();

    // Thread 0 reduces to final result
    if (threadIdx.x == 0)
    {
        float totalSum = 0.0;
        size_t setElements = (blockDim.x < N ? blockDim.x : N);
        
        for (int i = 0; i < setElements; i++)
        {
            totalSum += pData[i];
        }
        
        pData[0] = totalSum;  // Final result
    }
}
```

**Key Issue**: This kernel overwrites the input array `pData` with intermediate results, making subsequent runs produce different results.

### Without Checkpoints

```cpp
// Initialize array with all 1.0 values
for (size_t i = 0; i < elements; i++) {
    pHostA[i] = 1.0;
}
cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice);

// Run kernel multiple times
for (int repeat = 0; repeat < 3; repeat++) {
    Reduce<<<1, 64>>>(pDeviceA, elements);
    
    float result;
    cudaMemcpy(&result, pDeviceA, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Iteration %d: result = %f\n", repeat + 1, result);
}
```

**Output**:
```
Iteration 1: result = 1048576.000000  // Correct sum of 1M ones
Iteration 2: result = 64.000000       // Wrong! Input was modified
Iteration 3: result = 1.000000        // Even more wrong!
```

### With Checkpoints

```cpp
// Configure checkpoint
CUpti_Checkpoint cp = { CUpti_Checkpoint_STRUCT_SIZE };
cp.ctx = context;
cp.optimizations = 1;

float expected;

for (int repeat = 0; repeat < 3; repeat++) {
    // Save or restore checkpoint
    if (repeat == 0) {
        CUPTI_API_CALL(cuptiCheckpointSave(&cp));
    } else {
        CUPTI_API_CALL(cuptiCheckpointRestore(&cp));
    }
    
    // Run kernel with identical initial conditions
    Reduce<<<1, 64>>>(pDeviceA, elements);
    
    float result;
    cudaMemcpy(&result, pDeviceA, sizeof(float), cudaMemcpyDeviceToHost);
    
    if (repeat == 0) {
        expected = result;  // Save expected result
    }
    
    printf("Iteration %d: result = %f\n", repeat + 1, result);
    
    // Verify reproducibility
    if (result != expected) {
        printf("ERROR: Inconsistent result!\n");
        exit(1);
    }
}
```

**Output**:
```
Iteration 1: result = 1048576.000000  // Correct result
Iteration 2: result = 1048576.000000  // Same result!
Iteration 3: result = 1048576.000000  // Consistent!
```

## Building and Running

### Prerequisites

- CUDA Toolkit with CUPTI support
- C++ compiler compatible with CUDA
- GPU with compute capability 3.5 or higher

### Build Process

```bash
cd checkpoint_kernels
make
```

### Execution

```bash
./checkpoint_kernels
```

## Advanced Checkpoint Techniques

### Checkpoint Optimization

```cpp
// Enable optimizations for better performance
CUpti_Checkpoint cp = { CUpti_Checkpoint_STRUCT_SIZE };
cp.ctx = context;
cp.optimizations = 1;  // Enable all optimizations

// Alternative: Disable optimizations for debugging
cp.optimizations = 0;  // Slower but more thorough
```

### Selective Memory Checkpointing

```cpp
class SelectiveCheckpoint {
private:
    std::vector<CUpti_Checkpoint> checkpoints;
    std::vector<void*> criticalPointers;

public:
    void addCriticalMemory(void* ptr, size_t size) {
        criticalPointers.push_back(ptr);
        // Configure checkpoint for specific memory regions
    }
    
    void saveSelectiveState() {
        for (auto& cp : checkpoints) {
            CUPTI_API_CALL(cuptiCheckpointSave(&cp));
        }
    }
    
    void restoreSelectiveState() {
        for (auto& cp : checkpoints) {
            CUPTI_API_CALL(cuptiCheckpointRestore(&cp));
        }
    }
};
```

### Checkpoint-Based Debugging

```cpp
class CheckpointDebugger {
private:
    CUpti_Checkpoint debugCheckpoint;
    std::vector<float> expectedResults;

public:
    void setDebugPoint(CUcontext context) {
        debugCheckpoint = { CUpti_Checkpoint_STRUCT_SIZE };
        debugCheckpoint.ctx = context;
        debugCheckpoint.optimizations = 0;  // Full state capture
        
        CUPTI_API_CALL(cuptiCheckpointSave(&debugCheckpoint));
    }
    
    bool validateReproducibility(KernelFunction kernel, void* args) {
        // Run kernel multiple times and verify identical results
        std::vector<float> results;
        
        for (int run = 0; run < 5; run++) {
            if (run > 0) {
                CUPTI_API_CALL(cuptiCheckpointRestore(&debugCheckpoint));
            }
            
            kernel(args);
            
            float result = extractResult(args);
            results.push_back(result);
            
            if (run > 0 && results[run] != results[0]) {
                printf("Non-deterministic behavior detected at run %d\n", run);
                return false;
            }
        }
        
        printf("Kernel behavior is reproducible\n");
        return true;
    }
};
```

## Performance Considerations

### Checkpoint Overhead

```cpp
class CheckpointProfiler {
public:
    struct ProfileData {
        double saveTime;
        double restoreTime;
        size_t memorySize;
        double overhead;
    };
    
    ProfileData profileCheckpoint(CUpti_Checkpoint& cp) {
        ProfileData profile;
        
        // Measure save time
        auto start = std::chrono::high_resolution_clock::now();
        CUPTI_API_CALL(cuptiCheckpointSave(&cp));
        auto end = std::chrono::high_resolution_clock::now();
        
        profile.saveTime = std::chrono::duration<double>(end - start).count();
        
        // Measure restore time
        start = std::chrono::high_resolution_clock::now();
        CUPTI_API_CALL(cuptiCheckpointRestore(&cp));
        end = std::chrono::high_resolution_clock::now();
        
        profile.restoreTime = std::chrono::duration<double>(end - start).count();
        
        // Estimate memory size (implementation-dependent)
        profile.memorySize = estimateCheckpointSize(cp);
        profile.overhead = (profile.saveTime + profile.restoreTime) * 100;
        
        return profile;
    }
};
```

### Optimization Strategies

1. **Minimize Checkpoint Frequency**: Only checkpoint when necessary
2. **Use Selective Checkpointing**: Only save critical memory regions
3. **Enable Optimizations**: Use `cp.optimizations = 1` for better performance
4. **Batch Operations**: Group multiple kernel calls between checkpoints

## Real-World Use Cases

### Scientific Computing

```cpp
class IterativeSolver {
private:
    CUpti_Checkpoint convergenceCheckpoint;
    float* deviceVector;
    
public:
    void solveProblem() {
        // Save initial state for potential restart
        CUPTI_API_CALL(cuptiCheckpointSave(&convergenceCheckpoint));
        
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // Run potentially destructive iteration
            iterativeStep<<<blocks, threads>>>(deviceVector, size);
            
            if (checkConvergence(deviceVector)) {
                printf("Converged after %d iterations\n", iteration);
                break;
            }
            
            // If diverging, restart from checkpoint
            if (isDiverging(deviceVector)) {
                printf("Restarting from checkpoint\n");
                CUPTI_API_CALL(cuptiCheckpointRestore(&convergenceCheckpoint));
                adjustParameters();  // Try different parameters
            }
        }
    }
};
```

### Machine Learning Training

```cpp
class TrainingCheckpoint {
private:
    CUpti_Checkpoint epochCheckpoint;
    std::vector<float> lossHistory;

public:
    void trainWithCheckpoints(Model& model, Dataset& data) {
        for (int epoch = 0; epoch < totalEpochs; epoch++) {
            // Save state at beginning of epoch
            CUPTI_API_CALL(cuptiCheckpointSave(&epochCheckpoint));
            
            // Train one epoch
            float loss = trainEpoch(model, data);
            lossHistory.push_back(loss);
            
            // If loss exploded, restore and try different learning rate
            if (loss > explosionThreshold) {
                printf("Loss explosion detected, restoring checkpoint\n");
                CUPTI_API_CALL(cuptiCheckpointRestore(&epochCheckpoint));
                
                model.reduceLearningRate();
                lossHistory.pop_back();  // Remove bad result
                epoch--;  // Retry this epoch
            }
        }
    }
};
```

## Error Handling and Best Practices

### Robust Checkpoint Management

```cpp
class CheckpointManager {
private:
    std::vector<CUpti_Checkpoint> activeCheckpoints;
    
public:
    ~CheckpointManager() {
        // Ensure all checkpoints are cleaned up
        for (auto& cp : activeCheckpoints) {
            CUPTI_API_CALL(cuptiCheckpointFree(&cp));
        }
    }
    
    CUpti_Checkpoint* createCheckpoint(CUcontext context) {
        CUpti_Checkpoint cp = { CUpti_Checkpoint_STRUCT_SIZE };
        cp.ctx = context;
        cp.optimizations = 1;
        
        activeCheckpoints.push_back(cp);
        return &activeCheckpoints.back();
    }
    
    void validateCheckpoint(const CUpti_Checkpoint& cp) {
        // Verify checkpoint is valid before use
        if (cp.ctx == nullptr) {
            throw std::runtime_error("Invalid checkpoint context");
        }
        
        // Additional validation logic
    }
};
```

### Common Pitfalls to Avoid

1. **Forgetting to Free Checkpoints**: Always call `cuptiCheckpointFree()`
2. **Context Mismatches**: Ensure checkpoint context matches current context
3. **Incomplete State Capture**: Some GPU state may not be captured
4. **Performance Impact**: Checkpoints have overhead, use judiciously
5. **Memory Pressure**: Large checkpoints can consume significant memory

## Integration with Testing Frameworks

### Unit Testing with Checkpoints

```cpp
class CheckpointTest {
public:
    void testKernelReproducibility() {
        // Setup test data
        float* testData = setupTestData();
        
        // Create checkpoint
        CUpti_Checkpoint cp = createCheckpoint();
        
        // Run test multiple times
        for (int run = 0; run < 10; run++) {
            if (run > 0) {
                CUPTI_API_CALL(cuptiCheckpointRestore(&cp));
            }
            
            testKernel<<<1, 64>>>(testData, size);
            
            // Verify result consistency
            float result = extractResult(testData);
            ASSERT_EQ(result, expectedResult);
        }
        
        CUPTI_API_CALL(cuptiCheckpointFree(&cp));
    }
};
```

## Next Steps

- Experiment with different types of kernels to understand when checkpoints are needed
- Implement checkpoint-based debugging in your own applications
- Explore checkpoint optimizations for your specific use cases
- Combine checkpoints with other CUPTI profiling tools for comprehensive analysis
- Consider integrating checkpoint validation into your testing workflow

The checkpoint API is a powerful tool for ensuring reproducible GPU computations and can significantly improve the reliability of CUDA applications that modify their input data. 