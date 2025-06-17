# CUPTI SASS Source Mapping Tutorial

## Introduction

When optimizing CUDA kernels, understanding how your source code translates to actual GPU assembly instructions can provide powerful insights. SASS (Streaming Assembly) is the native assembly language executed by NVIDIA GPUs, and mapping between your CUDA C/C++ code and the resulting SASS instructions can reveal optimization opportunities that aren't apparent at the source level. This tutorial demonstrates how to use CUPTI to extract SASS code for your kernels and map it back to the original source code, helping you understand the relationship between your high-level code and the instructions that actually execute on the GPU.

## What You'll Learn

- How to extract SASS assembly code for CUDA kernels
- Mapping SASS instructions back to source code lines
- Interpreting SASS to understand instruction-level behavior
- Identifying optimization opportunities by analyzing SASS
- Using SASS information to make informed optimization decisions

## Understanding CUDA Compilation

When you compile CUDA code, it goes through several stages:

1. **CUDA C/C++**: Your high-level source code
2. **PTX**: An intermediate representation (Parallel Thread Execution)
3. **SASS**: The final machine code executed by the GPU

Each stage represents a different level of abstraction, and understanding the final SASS code can provide insights that aren't visible at the source level.

## Code Walkthrough

### 1. Loading a CUDA Module

First, we need to load a CUDA module to access its code:

```cpp
CUmodule module;
CUfunction function;

// Load the module
DRIVER_API_CALL(cuModuleLoad(&module, "kernel.cubin"));

// Get the kernel function
DRIVER_API_CALL(cuModuleGetFunction(&function, module, "vectorAdd"));
```

This code:
1. Loads a compiled CUDA binary (cubin) file
2. Gets a handle to a specific kernel function within that module

### 2. Extracting SASS Code

Next, we extract the SASS code for the kernel:

```cpp
// Get the function's code
CUdeviceptr code;
size_t codeSize;
DRIVER_API_CALL(cuFuncGetAttribute(&code, CU_FUNC_ATTRIBUTE_CODE, function));
DRIVER_API_CALL(cuFuncGetAttribute(&codeSize, CU_FUNC_ATTRIBUTE_BINARY_SIZE, function));

// Allocate memory for the code
unsigned char *sassCode = (unsigned char *)malloc(codeSize);
if (!sassCode) {
    fprintf(stderr, "Failed to allocate memory for SASS code\n");
    return -1;
}

// Copy the code from device memory
DRIVER_API_CALL(cuMemcpyDtoH(sassCode, code, codeSize));
```

This code:
1. Gets the device pointer to the function's code and its size
2. Allocates memory to hold the SASS code
3. Copies the code from device memory to host memory

### 3. Disassembling SASS

Now we disassemble the binary SASS code into a human-readable format:

```cpp
// Create a disassembler
CUpti_Activity_DisassembleData disassembleData;
memset(&disassembleData, 0, sizeof(disassembleData));
disassembleData.size = sizeof(disassembleData);
disassembleData.cubin = sassCode;
disassembleData.cubinSize = codeSize;
disassembleData.function = (const char *)function;

// Disassemble the code
CUPTI_CALL(cuptiActivityDisassembleKernel(&disassembleData));

// Get the disassembled SASS
const char *sassText = disassembleData.sass;
```

This code:
1. Sets up a structure for disassembly
2. Calls CUPTI to disassemble the kernel
3. Gets the resulting SASS text

### 4. Mapping SASS to Source Code

To map SASS instructions to source code, we use CUPTI's line information API:

```cpp
// Get the number of functions in the module
uint32_t numFunctions = 0;
CUPTI_CALL(cuptiModuleGetNumFunctions(module, &numFunctions));

// Get the function IDs
CUpti_ModuleResourceData *functionIds = 
    (CUpti_ModuleResourceData *)malloc(numFunctions * sizeof(CUpti_ModuleResourceData));
CUPTI_CALL(cuptiModuleGetFunctions(module, numFunctions, functionIds));

// For each function
for (uint32_t i = 0; i < numFunctions; i++) {
    // Check if this is our target function
    if (strcmp(functionIds[i].resourceName, "vectorAdd") == 0) {
        // Get line information
        uint32_t numLines = 0;
        CUPTI_CALL(cuptiGetNumLines(functionIds[i].function, &numLines));
        
        // Allocate memory for line information
        CUpti_LineInfo *lineInfo = 
            (CUpti_LineInfo *)malloc(numLines * sizeof(CUpti_LineInfo));
        
        // Get the line information
        CUPTI_CALL(cuptiGetLineInfo(functionIds[i].function, numLines, lineInfo));
        
        // Process line information
        for (uint32_t j = 0; j < numLines; j++) {
            printf("SASS instruction at offset 0x%x maps to %s:%d\n",
                   lineInfo[j].pcOffset, lineInfo[j].fileName, lineInfo[j].lineNumber);
        }
        
        free(lineInfo);
    }
}

free(functionIds);
```

This code:
1. Gets the list of functions in the module
2. Finds our target function
3. Gets line information for that function
4. Maps each SASS instruction offset to a source file and line number

### 5. Creating a Source-Annotated SASS Listing

Now we combine the SASS code with source line information:

```cpp
void printSourceAnnotatedSass(const char *sassText, CUpti_LineInfo *lineInfo, uint32_t numLines)
{
    // Parse the SASS text
    char *sassCopy = strdup(sassText);
    char *line = strtok(sassCopy, "\n");
    
    int currentSourceLine = -1;
    const char *currentFileName = NULL;
    
    // Process each line of SASS
    while (line != NULL) {
        // Extract the instruction offset
        unsigned int offset;
        if (sscanf(line, "/*%x*/", &offset) == 1) {
            // Find the source line for this offset
            for (uint32_t i = 0; i < numLines; i++) {
                if (lineInfo[i].pcOffset == offset) {
                    // If we've moved to a new source line, print it
                    if (currentSourceLine != lineInfo[i].lineNumber || 
                        currentFileName != lineInfo[i].fileName) {
                        currentSourceLine = lineInfo[i].lineNumber;
                        currentFileName = lineInfo[i].fileName;
                        
                        // Read the source file and get the line
                        char sourceLine[1024];
                        FILE *sourceFile = fopen(currentFileName, "r");
                        if (sourceFile) {
                            for (int j = 0; j < currentSourceLine; j++) {
                                if (!fgets(sourceLine, sizeof(sourceLine), sourceFile)) {
                                    break;
                                }
                            }
                            fclose(sourceFile);
                            // Remove newline
                            sourceLine[strcspn(sourceLine, "\n")] = 0;
                            
                            printf("\nSource Line %d: %s\n", currentSourceLine, sourceLine);
                        }
                    }
                    break;
                }
            }
        }
        
        // Print the SASS instruction
        printf("    %s\n", line);
        
        // Get the next line
        line = strtok(NULL, "\n");
    }
    
    free(sassCopy);
}
```

This function:
1. Parses the SASS text line by line
2. Extracts the instruction offset from each line
3. Finds the corresponding source line for that offset
4. Prints the source line followed by the SASS instructions

### 6. Sample Kernel for Analysis

Here's a simple vector addition kernel we'll analyze:

```cpp
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

This kernel:
1. Calculates a global thread index
2. Checks if the index is within bounds
3. Performs a simple vector addition

## Running the Tutorial

1. Build the sample:
   ```bash
   make
   ```

2. Run the SASS source mapping example:
   ```bash
   ./sass_source_map
   ```

## Understanding the Output

When you run the SASS source mapping example, you'll see output similar to this:

```
SASS Disassembly for kernel 'vectorAdd':

Source Line 42: int i = blockIdx.x * blockDim.x + threadIdx.x;
    /*0008*/    MOV R1, c[0x0][0x44];      /* Source Line 42 */
    /*0010*/    S2R R0, SR_CTAID.X;        /* Source Line 42 */
    /*0018*/    S2R R3, SR_TID.X;          /* Source Line 42 */
    /*0020*/    IMAD R0, R0, c[0x0][0x28], R3; /* Source Line 42 */

Source Line 43: if (i < n) {
    /*0028*/    ISETP.GE.AND P0, PT, R0, R1, PT; /* Source Line 43 */
    /*0030*/    @P0 EXIT;                  /* Source Line 43 */

Source Line 44:     c[i] = a[i] + b[i];
    /*0038*/    IMUL R3, R0, 0x4;          /* Source Line 44 */
    /*0040*/    IMAD R2, R0, 0x4, c[0x0][0x140]; /* Source Line 44 */
    /*0048*/    IMAD R1, R0, 0x4, c[0x0][0x148]; /* Source Line 44 */
    /*0050*/    IMAD R0, R0, 0x4, c[0x0][0x150]; /* Source Line 44 */
    /*0058*/    LDG R2, [R2];              /* Source Line 44 */
    /*0060*/    LDG R1, [R1];              /* Source Line 44 */
    /*0068*/    IADD R1, R1, R2;           /* Source Line 44 */
    /*0070*/    STG [R0], R1;              /* Source Line 44 */

Source Line 45: }
    /*0078*/    EXIT;                      /* Source Line 45 */

Performance Analysis:
  Line 42 (Thread index calculation): 4 instructions (20% of kernel instructions)
  Line 43 (Bounds check): 2 instructions (10% of kernel instructions)
  Line 44 (Array access and computation): 8 instructions (70% of kernel instructions)
  Line 45 (Kernel exit): 1 instruction
```

Let's analyze this output:

### Thread Index Calculation (Line 42)

```
/*0008*/    MOV R1, c[0x0][0x44];      /* Source Line 42 */
/*0010*/    S2R R0, SR_CTAID.X;        /* Source Line 42 */
/*0018*/    S2R R3, SR_TID.X;          /* Source Line 42 */
/*0020*/    IMAD R0, R0, c[0x0][0x28], R3; /* Source Line 42 */
```

These instructions:
1. `MOV R1, c[0x0][0x44]`: Load the value of `n` into register R1
2. `S2R R0, SR_CTAID.X`: Load the block index into R0
3. `S2R R3, SR_TID.X`: Load the thread index into R3
4. `IMAD R0, R0, c[0x0][0x28], R3`: Calculate `blockIdx.x * blockDim.x + threadIdx.x`

### Bounds Check (Line 43)

```
/*0028*/    ISETP.GE.AND P0, PT, R0, R1, PT; /* Source Line 43 */
/*0030*/    @P0 EXIT;                  /* Source Line 43 */
```

These instructions:
1. `ISETP.GE.AND P0, PT, R0, R1, PT`: Set predicate P0 if `i >= n`
2. `@P0 EXIT`: Exit the kernel if P0 is true (i.e., if `i >= n`)

### Vector Addition (Line 44)

```
/*0038*/    IMUL R3, R0, 0x4;          /* Source Line 44 */
/*0040*/    IMAD R2, R0, 0x4, c[0x0][0x140]; /* Source Line 44 */
/*0048*/    IMAD R1, R0, 0x4, c[0x0][0x148]; /* Source Line 44 */
/*0050*/    IMAD R0, R0, 0x4, c[0x0][0x150]; /* Source Line 44 */
/*0058*/    LDG R2, [R2];              /* Source Line 44 */
/*0060*/    LDG R1, [R1];              /* Source Line 44 */
/*0068*/    IADD R1, R1, R2;           /* Source Line 44 */
/*0070*/    STG [R0], R1;              /* Source Line 44 */
```

These instructions:
1. `IMUL R3, R0, 0x4`: Multiply index by 4 (size of float)
2. `IMAD R2, R0, 0x4, c[0x0][0x140]`: Calculate address of `a[i]`
3. `IMAD R1, R0, 0x4, c[0x0][0x148]`: Calculate address of `b[i]`
4. `IMAD R0, R0, 0x4, c[0x0][0x150]`: Calculate address of `c[i]`
5. `LDG R2, [R2]`: Load `a[i]` into R2
6. `LDG R1, [R1]`: Load `b[i]` into R1
7. `IADD R1, R1, R2`: Add R1 and R2, store result in R1
8. `STG [R0], R1`: Store result in `c[i]`

### Kernel Exit (Line 45)

```
/*0078*/    EXIT;                      /* Source Line 45 */
```

This instruction:
1. `EXIT`: Exit the kernel

## Interpreting SASS for Optimization

### Memory Access Patterns

In the SASS for line 44, we see:

```
/*0058*/    LDG R2, [R2];              /* Source Line 44 */
/*0060*/    LDG R1, [R1];              /* Source Line 44 */
```

These are global memory loads. The `LDG` instruction loads from global memory. For optimal performance:

- Adjacent threads should access adjacent memory locations
- Memory accesses should be aligned to 128-byte boundaries
- Coalesced memory access is critical for performance

### Instruction Mix

Looking at the instruction mix:

- 4 instructions for thread index calculation (20%)
- 2 instructions for bounds checking (10%)
- 8 instructions for the actual computation (70%)

This tells us:
- The kernel has a reasonable ratio of computation to overhead
- Most instructions are dedicated to the actual vector addition
- The thread index calculation and bounds checking are relatively efficient

### Register Usage

The SASS shows register usage:

- R0: Used for thread index and later for the address of `c[i]`
- R1: Initially holds `n`, later holds `b[i]` and the final result
- R2: Holds the address of `a[i]` and later the value of `a[i]`
- R3: Initially holds thread ID, later used for byte offset calculation

Register usage is efficient with good reuse of registers.

## Advanced SASS Analysis

### Instruction Throughput

Different SASS instructions have different throughput:

- `MOV`, `S2R`: Fast register operations
- `IMAD`, `IMUL`: Integer arithmetic (medium throughput)
- `LDG`, `STG`: Global memory operations (slow, high latency)
- `ISETP`: Predicate operations (medium throughput)

In performance-critical kernels, minimizing the use of slow instructions is important.

### Predicated Execution

The SASS shows predicated execution:

```
/*0030*/    @P0 EXIT;                  /* Source Line 43 */
```

This uses the predicate register P0 to conditionally execute the EXIT instruction. Predication can avoid branch divergence but has its own costs.

### Memory Address Calculation

The SASS shows memory address calculations:

```
/*0040*/    IMAD R2, R0, 0x4, c[0x0][0x140]; /* Source Line 44 */
```

This calculates `base_address + i * 4` using a multiply-add instruction. The compiler has optimized this calculation.

## Optimization Strategies Based on SASS

### 1. Memory Coalescing

If the SASS shows many uncoalesced memory accesses, consider:
- Reorganizing your data structures
- Using shared memory as a staging area
- Adjusting your thread block dimensions

### 2. Instruction Reduction

If certain source lines generate many instructions:
- Simplify complex expressions
- Use intrinsic functions when appropriate
- Consider algorithmic changes

### 3. Register Pressure

If the SASS shows high register usage:
- Break complex functions into smaller ones
- Reduce the number of variables in flight
- Consider using shared memory instead of registers for some data

## Next Steps

- Apply SASS analysis to your own CUDA kernels
- Look for patterns in the generated code that might indicate inefficiencies
- Compare SASS across different compiler optimization levels
- Use SASS insights to guide source-level optimizations 