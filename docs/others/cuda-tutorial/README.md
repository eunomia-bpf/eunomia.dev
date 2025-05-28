# basic-cuda-tutorial

You can find the code in <https://github.com/eunomia-bpf/basic-cuda-tutorial>

A collection of CUDA programming examples to learn GPU programming with NVIDIA CUDA.

Make sure to change the gpu architecture `sm_61` to your own gpu architecture in Makefile

## Examples and tutorials

- **01-vector-addition.cu** and [01-vector-addition.md](01-vector-addition.md): Introduction to CUDA programming with a vector addition example
- **02-ptx-assembly.cu** and [02-ptx-assembly.md](02-ptx-assembly.md): Demonstration of CUDA PTX inline assembly with a vector multiplication example
- **03-gpu-programming-methods.cu** and [03-gpu-programming-methods.md](03-gpu-programming-methods.md): Comprehensive comparison of GPU programming methods including CUDA, PTX, Thrust, Unified Memory, Shared Memory, CUDA Streams, and Dynamic Parallelism using matrix multiplication
- **04-gpu-architecture.cu** and [04-gpu-architecture.md](04-gpu-architecture.md): Detailed exploration of GPU organization hierarchy including hardware architecture, thread/block/grid structure, memory hierarchy, and execution model
- **05-neural-network.cu** and [05-neural-network.md](05-neural-network.md): Implementing a basic neural network forward pass on GPU with CUDA
- **06-cnn-convolution.cu** and [06-cnn-convolution.md](06-cnn-convolution.md): GPU-accelerated convolution operations for CNN with shared memory optimization
- **07-attention-mechanism.cu** and [07-attention-mechanism.md](07-attention-mechanism.md): CUDA implementation of attention mechanism for transformer models
- **08-profiling-tracing.cu** and [08-profiling-tracing.md](08-profiling-tracing.md): Profiling and tracing CUDA applications with CUDA Events, NVTX, and CUPTI for performance optimization
- **09-gpu-extension.cu** and [09-gpu-extension.md](09-gpu-extension.md): GPU application extension mechanisms for modifying behavior without source code changes, including API interception, memory management, kernel optimization, and error resilience
- **10-cpu-gpu-profiling-boundaries.cu** and [10-cpu-gpu-profiling-boundaries.md](10-cpu-gpu-profiling-boundaries.md): Advanced GPU kernel instrumentation techniques demonstrating fine-grained internal timing, divergent path analysis, dynamic workload profiling, and adaptive algorithm selection within CUDA kernels
- **11-fine-grained-gpu-modifications.cu** and [11-fine-grained-gpu-modifications.md](11-fine-grained-gpu-modifications.md): Fine-grained GPU code customizations including data structure layout optimization, warp-level primitives, memory access patterns, kernel fusion, and dynamic execution path selection
- **12-advanced-gpu-customizations.cu** and [12-advanced-gpu-customizations.md](12-advanced-gpu-customizations.md): Advanced GPU customization techniques including thread divergence mitigation, register usage optimization, mixed precision computation, persistent threads for load balancing, and warp specialization patterns
- **13-low-latency-gpu-packet-processing.cu** and [13-low-latency-gpu-packet-processing.md](13-low-latency-gpu-packet-processing.md): Techniques for minimizing latency in GPU-based network packet processing, including pinned memory, zero-copy memory, stream pipelining, persistent kernels, and CUDA Graphs for real-time network applications

Each tutorial includes comprehensive documentation explaining the concepts, implementation details, and optimization techniques used in ML/AI workloads on GPUs.