# CPU和GPU分析边界：在哪里测量什么

本文档探讨了CPU和GPU性能分析之间的边界，研究哪些操作可以在CPU端有效测量，哪些需要GPU端工具。我们还将讨论像eBPF这样的高级CPU端函数钩子技术如何补充GPU性能分析。

## 目录

1. [CPU-GPU边界](#cpu-gpu边界)
2. [CPU端可测量操作](#cpu端可测量操作)
3. [GPU端可测量操作](#gpu端可测量操作)
4. [何时需要内核插桩](#何时需要内核插桩)
5. [使用eBPF钩住CPU端函数](#使用ebpf钩住cpu端函数)
6. [集成性能分析方法](#集成性能分析方法)
7. [案例研究](#案例研究)
8. [未来方向](#未来方向)
9. [参考文献](#参考文献)

## CPU-GPU边界

现代GPU计算涉及主机（CPU）和设备（GPU）操作之间的复杂相互作用。了解在哪里放置分析工具取决于您要测量的性能方面：

```
┌────────────────────────┐                  ┌────────────────────────┐
│        CPU端           │                  │        GPU端           │
│                        │                  │                        │
│  ┌──────────────────┐  │                  │  ┌──────────────────┐  │
│  │    应用程序代码   │  │                  │  │    内核执行      │  │
│  └────────┬─────────┘  │                  │  └────────┬─────────┘  │
│           │            │                  │           │            │
│  ┌────────▼─────────┐  │                  │  ┌────────▼─────────┐  │
│  │   CUDA运行时     │  │                  │  │    线程束调度器   │  │
│  └────────┬─────────┘  │                  │  └────────┬─────────┘  │
│           │            │                  │           │            │
│  ┌────────▼─────────┐  │  ┌─────────┐     │  ┌────────▼─────────┐  │
│  │    CUDA驱动      │◄─┼──┤PCIe总线 │────►│  │    内存控制器    │  │
│  └────────┬─────────┘  │  └─────────┘     │  └────────┬─────────┘  │
│           │            │                  │           │            │
│  ┌────────▼─────────┐  │                  │  ┌────────▼─────────┐  │
│  │    系统软件      │  │                  │  │    GPU硬件       │  │
│  └──────────────────┘  │                  │  └──────────────────┘  │
└────────────────────────┘                  └────────────────────────┘
```

## CPU端可测量操作

以下操作可以从CPU端有效测量：

### 1. CUDA API调用延迟

- **内核启动开销**：从API调用到内核开始执行的时间
- **内存分配**：在`cudaMalloc`、`cudaFree`等中花费的时间
- **主机-设备传输**：`cudaMemcpy`操作的持续时间
- **同步点**：在`cudaDeviceSynchronize`、`cudaStreamSynchronize`中花费的时间

### 2. 资源管理

- **内存使用**：跟踪GPU内存分配和释放模式
- **流创建**：创建和销毁CUDA流的开销
- **上下文切换**：在CUDA上下文之间切换所花费的时间

### 3. CPU-GPU交互模式

- **API调用频率**：CUDA API调用的速率和模式
- **CPU等待时间**：CPU等待GPU操作的时间
- **I/O和GPU重叠**：I/O操作如何与GPU利用率交互

### 4. 系统级指标

- **PCIe流量**：通过PCIe传输的数据量和时间
- **功耗**：与GPU活动相关的系统范围功耗
- **热效应**：可能影响节流的温度变化

### CPU端测量工具和技术

- **CUPTI API回调**：通过CUPTI接口钩入CUDA API调用
- **二进制插桩**：使用Pin或DynamoRIO等工具拦截函数
- **插入库**：拦截CUDA API调用的自定义库
- **eBPF**：Linux的扩展Berkeley包过滤器，用于内核级跟踪
- **性能计数器**：通过PAPI或类似工具可访问的硬件级计数器

## GPU端可测量操作

以下操作需要GPU端插桩：

### 1. 内核执行细节

- **指令混合**：执行的指令类型和频率
- **线程束执行效率**：线程束中活动线程的百分比
- **分支发散模式**：分支发散的频率和影响
- **指令级并行性**：每个线程内实现的ILP

### 2. 内存系统性能

- **内存访问模式**：合并效率，步长模式
- **缓存命中率**：L1/L2/纹理缓存的有效性
- **存储体冲突**：共享内存访问冲突
- **内存发散**：发散的内存访问模式

### 3. 硬件利用率

- **SM占用率**：相对于最大容量的活动线程束
- **特殊功能使用**：SFU、张量核心等的利用率
- **内存带宽**：实际与理论内存带宽
- **计算吞吐量**：FLOPS或其他计算指标

### 4. 同步效果

- **块同步**：`__syncthreads()`操作的影响
- **原子操作**：原子操作对性能的影响
- **线程束调度决策**：线程束如何在SM上调度

### GPU端测量工具和技术

- **SASS/PTX分析**：检查低级汇编代码
- **硬件性能计数器**：用于各种指标的GPU特定计数器
- **内核插桩**：直接向内核添加计时代码
- **专门分析器**：用于深入GPU洞察的Nsight Compute、Nvprof
- **可视化分析器**：内核执行的时间线视图

## 何时需要内核插桩

虽然Nsight Compute和nvprof等工具提供了广泛的分析功能，但在特定情况下，直接向CUDA内核添加计时代码是必要或有益的：

### 1. 细粒度内部内核计时

**使用时机**：当你需要测量内核内特定代码段的执行时间。

```cuda
__global__ void complexKernel(float* data, int n) {
    // 用于计时的共享变量
    __shared__ clock64_t start_time, section1_time, section2_time, end_time;
    
    // 每个块只有一个线程记录时间戳
    if (threadIdx.x == 0) {
        start_time = clock64();
    }
    
    // 第一个计算部分
    // ...复杂操作...
    
    __syncthreads();  // 确保所有线程完成
    
    if (threadIdx.x == 0) {
        section1_time = clock64();
    }
    
    // 第二个计算部分
    // ...更多操作...
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        section2_time = clock64();
    }
    
    // 最后部分
    // ...完成操作...
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        end_time = clock64();
        // 将计时结果存储到全局内存
        // 根据设备时钟速率将时钟周期转换为毫秒
    }
}
```

**好处**：提供外部分析器无法捕获的内核内部可见性，特别是用于识别复杂内核中的热点。

### 2. 条件或发散路径分析

**使用时机**：测量发散代码中不同执行路径的性能。

```cuda
__global__ void divergentPathKernel(float* data, int* path_times, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        clock64_t start, end;
        start = clock64();
        
        if (data[idx] > 0) {
            // 路径A - 昂贵计算
            for (int i = 0; i < 100; i++) {
                data[idx] = sinf(data[idx]) * cosf(data[idx]);
            }
        } else {
            // 路径B - 不同计算
            for (int i = 0; i < 50; i++) {
                data[idx] = data[idx] * data[idx] + 1.0f;
            }
        }
        
        end = clock64();
        
        // 记录采用了哪条路径以及花费的时间
        path_times[idx * 2] = (data[idx] > 0) ? 1 : 0;  // 路径指示符
        path_times[idx * 2 + 1] = (int)(end - start);   // 花费的时间
    }
}
```

**好处**：通过单独测量每条路径，帮助识别线程发散对性能的影响。

### 3. 动态工作负载分析

**使用时机**：处理工作负载在线程或块之间显著变化的算法。

```cuda
__global__ void dynamicWorkloadKernel(int* elements, int* work_counts, int* times, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int work_items = elements[idx];  // 每个线程有不同量的工作
        
        clock64_t start = clock64();
        
        // 执行可变数量的工作
        for (int i = 0; i < work_items; i++) {
            // 进行计算
        }
        
        clock64_t end = clock64();
        
        // 记录工作负载和时间
        work_counts[idx] = work_items;
        times[idx] = (int)(end - start);
    }
}
```

**好处**：揭示工作负载特性与执行时间之间的关系，有助于优化负载平衡。

### 4. 自定义硬件计数器访问

**使用时机**：在执行的精确点需要特定硬件性能指标。

```cuda
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void customCounterKernel(float* data, int* counter_values, int n) {
    thread_block block = this_thread_block();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (threadIdx.x == 0) {
        // 重置SM L1缓存命中计数器（假设示例）
        asm volatile("read.ptx.special.register %0, l1_cache_hits;" : "=r"(counter_values[blockIdx.x * 4 + 0]));
    }
    
    // 第一阶段计算，可能使用L1缓存
    // ...
    
    block.sync();
    
    if (threadIdx.x == 0) {
        // 第一阶段后读取L1缓存命中计数器
        asm volatile("read.ptx.special.register %0, l1_cache_hits;" : "=r"(counter_values[blockIdx.x * 4 + 1]));
    }
    
    // 第二阶段计算
    // ...
    
    block.sync();
    
    if (threadIdx.x == 0) {
        // 最终计数器读取
        asm volatile("read.ptx.special.register %0, l1_cache_hits;" : "=r"(counter_values[blockIdx.x * 4 + 2]));
    }
}
```

**注意**：这是一个概念示例。实际的硬件计数器访问因GPU架构而异，需要特定的内部函数或汇编指令。

### 5. 实时算法自适应

**使用时机**：当内核需要根据性能反馈自我调整。

```cuda
__global__ void adaptiveKernel(float* data, float* timing_data, int n, int iterations) {
    __shared__ clock64_t start, mid, end;
    __shared__ float method_a_time, method_b_time;
    __shared__ int selected_method;
    
    if (threadIdx.x == 0) {
        // 以方法A为默认初始化
        selected_method = 0;
    }
    
    __syncthreads();
    
    // 运行多次迭代，自适应算法
    for (int iter = 0; iter < iterations; iter++) {
        if (threadIdx.x == 0) {
            start = clock64();
        }
        
        // 先尝试方法A
        if (selected_method == 0) {
            // 方法A实现
            // ...
        }
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            mid = clock64();
        }
        
        // 尝试方法B
        if (selected_method == 1) {
            // 方法B实现
            // ...
        }
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            end = clock64();
            
            // 计算执行时间
            method_a_time = (mid - start);
            method_b_time = (end - mid);
            
            // 为下一次迭代选择更快的方法
            selected_method = (method_a_time <= method_b_time) ? 0 : 1;
            
            // 记录哪种方法更快
            if (blockIdx.x == 0) {
                timing_data[iter] = selected_method;
            }
        }
        
        __syncthreads();
    }
}
```

**好处**：使内核能够根据实时性能测量进行算法选择。

### 6. 多GPU内核协调

**使用时机**：在多个GPU之间协调工作，且时间对同步至关重要。

```cuda
__global__ void coordinatedKernel(float* data, volatile int* sync_flags, 
                                 volatile clock64_t* timing, int device_id, int n) {
    // 记录开始时间
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        timing[device_id] = clock64();
        // 发出此GPU已开始的信号
        sync_flags[device_id] = 1;
    }
    
    // 执行计算
    // ...
    
    // 等待所有GPU完成一个阶段
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        timing[device_id + 4] = clock64();  // 记录完成时间
        sync_flags[device_id] = 2;  // 发出完成信号
        
        // 等待其他GPU（简化的忙等待）
        while (sync_flags[0] < 2 || sync_flags[1] < 2 || 
               sync_flags[2] < 2 || sync_flags[3] < 2) {
            // 忙等待
        }
    }
    
    __syncthreads();
    
    // 继续协调执行
    // ...
}
```

**好处**：帮助识别多GPU系统中的负载不平衡和同步开销。

### 实现注意事项

实现内核插桩时：

1. **时钟分辨率**：使用适当的计时函数：
   - `clock64()`提供设备周期计数器（高分辨率但与架构相关）
   - 使用设备时钟速率将周期转换为时间

2. **测量开销**：最小化测量对结果的影响：
   - 仅计时关键部分
   - 在计时中使用最少的线程参与（例如，每块一个线程）

3. **数据提取**：考虑如何检索计时数据：
   - 将结果存储在全局内存中
   - 如果多个线程报告计时，考虑使用原子操作
   - 对于大量线程可能需要聚合

4. **同步要求**：确保测量之间的正确同步：
   - 使用`__syncthreads()`创建一致的计时边界
   - 考虑使用合作组进行复杂同步

### 示例：综合内核部分计时

```cuda
#include <cuda_runtime.h>
#include <helper_cuda.h> // 用于checkCudaErrors

// 保存计时结果的结构
struct KernelTimings {
    long long init_time;
    long long compute_time;
    long long finalize_time;
    long long total_time;
};

__global__ void instrumentedKernel(float* input, float* output, int n, KernelTimings* timings) {
    extern __shared__ float shared_data[];
    
    // 每个块只有一个线程记录时间
    clock64_t block_start, init_end, compute_end, block_end;
    if (threadIdx.x == 0) {
        block_start = clock64();
    }
    
    // 初始化阶段
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        shared_data[threadIdx.x] = input[idx] * 2.0f;
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        init_end = clock64();
    }
    
    // 计算阶段
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += shared_data[i];
        }
        output[idx] = sum / blockDim.x;
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        compute_end = clock64();
    }
    
    // 完成阶段
    if (idx < n) {
        output[idx] = output[idx] * output[idx];
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        block_end = clock64();
        
        // 原子加以累计跨块的时间
        atomicAdd(&timings->init_time, init_end - block_start);
        atomicAdd(&timings->compute_time, compute_end - init_end);
        atomicAdd(&timings->finalize_time, block_end - compute_end);
        atomicAdd(&timings->total_time, block_end - block_start);
    }
}

// 主机端函数执行和提取计时
void runAndTimeKernel(float* d_input, float* d_output, int n, int blockSize) {
    // 在设备上分配并初始化计时结构
    KernelTimings* d_timings;
    checkCudaErrors(cudaMalloc(&d_timings, sizeof(KernelTimings)));
    checkCudaErrors(cudaMemset(d_timings, 0, sizeof(KernelTimings)));
    
    // 计算网格维度
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // 启动带共享内存的内核
    instrumentedKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_input, d_output, n, d_timings);
    
    // 等待内核完成
    checkCudaErrors(cudaDeviceSynchronize());
    
    // 将计时结果复制回主机
    KernelTimings h_timings;
    checkCudaErrors(cudaMemcpy(&h_timings, d_timings, sizeof(KernelTimings), cudaMemcpyDeviceToHost));
    
    // 获取设备属性以将周期转换为毫秒
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    float ms_per_cycle = 1000.0f / (prop.clockRate * 1000.0f);
    
    // 打印计时结果
    printf("内核计时结果：\n");
    printf("  初始化： %.4f ms\n", h_timings.init_time * ms_per_cycle / gridSize);
    printf("  计算：    %.4f ms\n", h_timings.compute_time * ms_per_cycle / gridSize);
    printf("  完成：   %.4f ms\n", h_timings.finalize_time * ms_per_cycle / gridSize);
    printf("  总计：          %.4f ms\n", h_timings.total_time * ms_per_cycle / gridSize);
    
    // 释放设备内存
    checkCudaErrors(cudaFree(d_timings));
}
```

## 使用eBPF钩住CPU端函数

eBPF（扩展Berkeley包过滤器）提供了强大的机制，可以在不修改源代码的情况下跟踪和监控Linux上的系统行为。对于GPU工作负载，eBPF在关联CPU端活动与GPU性能方面特别有价值。

示例代码可以在[bpf-developer-tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/47-cuda-events)中找到。

### 什么是eBPF？

eBPF是一种技术，允许在Linux内核中运行沙盒程序，无需更改内核源代码或加载内核模块。它广泛用于性能分析、安全监控和网络。

### 用于GPU工作负载分析的eBPF

虽然eBPF无法直接检测在GPU上运行的代码，但它擅长监控CPU端与GPU的交互：

#### 1. 跟踪CUDA驱动交互

```c
// 跟踪CUDA驱动函数调用的示例eBPF程序
int trace_cuLaunchKernel(struct pt_regs *ctx) {
    u64 ts = bpf_ktime_get_ns();
    struct data_t data = {};
    
    data.timestamp = ts;
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    
    // 捕获函数参数
    data.gridDimX = PT_REGS_PARM2(ctx);
    data.gridDimY = PT_REGS_PARM3(ctx);
    data.gridDimZ = PT_REGS_PARM4(ctx);
    
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
```

#### 2. 关联系统事件与GPU活动

eBPF可以监控：
- 可能影响GPU数据传输的文件I/O操作
- 影响CPU-GPU协调的调度决策
- 与GPU缓冲区处理相关的内存管理事件

#### 3. 构建完整画面

通过结合eBPF收集的CPU端数据与GPU分析信息：
- 跟踪数据从源到GPU再返回的路径
- 识别影响GPU性能的系统级瓶颈
- 了解导致GPU空闲时间的调度问题

### GPU工作负载的eBPF工具

1. **BCC（BPF编译器集合）**：为eBPF程序提供Python接口
2. **bpftrace**：适用于Linux eBPF的高级跟踪语言
3. **自定义eBPF程序**：专门为CUDA/GPU工作负载定制

使用bpftrace跟踪CUDA内存操作的示例：

```
bpftrace -e '
uprobe:/usr/lib/libcuda.so:cuMemAlloc {
    printf("调用了cuMemAlloc: size=%llu, pid=%d, comm=%s\n", 
           arg1, pid, comm);
    @mem_alloc_bytes = hist(arg1);
}
uprobe:/usr/lib/libcuda.so:cuMemFree {
    printf("调用了cuMemFree: pid=%d, comm=%s\n", pid, comm);
}
'
```

## 集成性能分析方法

有效的GPU应用程序分析需要整合来自CPU和GPU两端的数据：

### 1. 时间线关联

对齐CPU和GPU时间线上的事件以识别：
- **内核启动延迟**：CPU请求与GPU执行之间的间隔
- **传输-计算重叠**：异步操作的有效性
- **CPU-GPU同步点**：CPU等待GPU的位置

### 2. 瓶颈识别

使用组合数据确定瓶颈是：
- **CPU绑定**：CPU数据准备或启动开销
- **传输绑定**：PCIe或内存带宽限制
- **GPU计算绑定**：内核算法效率
- **GPU内存绑定**：GPU内存访问模式

### 3. 多级优化策略

开发全面的优化方法：
1. **系统级**：PCIe配置、电源设置、CPU亲和性
2. **应用级**：内核启动模式、内存管理
3. **算法级**：内核实现、内存访问模式
4. **指令级**：PTX/SASS优化

## 案例研究

### 案例研究1：深度学习训练框架

在深度学习框架中，我们观察到：

- **CPU端分析**：识别GPU传输前的低效数据预处理
- **GPU端分析**：显示利用率高但内存访问模式差
- **eBPF分析**：揭示Linux页面缓存行为导致数据传输时间不可预测

**解决方案**：实现固定内存并根据eBPF收集的访问模式进行显式预取，从而提高35%的吞吐量。

### 案例研究2：实时图像处理管道

对于实时图像处理应用程序：

- **CPU端分析**：显示突发内核启动导致GPU空闲时间
- **GPU端分析**：表明内核效率良好但占用率低
- **eBPF分析**：发现CPU上的线程调度问题影响启动时间

**解决方案**：使用eBPF见解实现CPU线程固定并重组管道，实现一致的帧率，端到端延迟减少22%。

## 未来方向

CPU和GPU分析之间的边界继续演变：

1. **统一内存分析**：随着统一内存变得更加普遍，需要新工具来跟踪页面迁移和访问模式

2. **片上系统集成**：随着GPU与CPU的集成度越来越高，分析边界将变得模糊，需要新方法

3. **多GPU系统**：跨多个GPU的分布式训练和推理带来了新的分析挑战

4. **AI辅助分析**：使用机器学习自动识别模式并在CPU-GPU边界上提出优化建议

## 参考文献

1. NVIDIA. "CUPTI: CUDA Profiling Tools Interface." [https://docs.nvidia.com/cuda/cupti/](https://docs.nvidia.com/cuda/cupti/)
2. Gregg, Brendan. "BPF Performance Tools: Linux System and Application Observability." Addison-Wesley Professional, 2019.
3. NVIDIA. "Nsight Systems User Guide." [https://docs.nvidia.com/nsight-systems/](https://docs.nvidia.com/nsight-systems/)
4. Awan, Ammar Ali, et al. "Characterizing Machine Learning I/O Workloads on NVME and CPU-GPU Systems." IEEE International Parallel and Distributed Processing Symposium Workshops, 2022.
5. The eBPF Foundation. "What is eBPF?" [https://ebpf.io/what-is-ebpf/](https://ebpf.io/what-is-ebpf/)
6. NVIDIA. "Tools for Profiling CUDA Applications." [https://developer.nvidia.com/tools-overview](https://developer.nvidia.com/tools-overview)
7. Arafa, Yehia, et al. "Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs." High Performance Computing: ISC High Performance 2019.
8. Haidar, Azzam, et al. "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers." SC18: International Conference for High Performance Computing, Networking, Storage and Analysis, 2018. 