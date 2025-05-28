NVCC = nvcc

# Auto-detect GPU architecture
ARCH := $(shell $(NVCC) -run ./detect_arch.cu 2>/dev/null || echo "sm_61")

NVCC_FLAGS = -O3 -arch=$(ARCH)
NVCC_DP_FLAGS = -O3 -arch=$(ARCH) -rdc=true
NVCC_PROF_FLAGS = -O3 -arch=$(ARCH) -lcupti -lnvToolsExt
NVCC_EXT_FLAGS = -O3 -arch=$(ARCH) -ldl -lpthread
NVCC_PTX_FLAGS = -std=c++11 -arch=$(ARCH)

# OpenCL compiler and flags
CC = gcc
OPENCL_FLAGS = -O3 -std=c99
# Try to detect OpenCL libraries
OPENCL_LIBS := $(shell pkg-config --libs OpenCL 2>/dev/null || echo "-lOpenCL")
OPENCL_INCLUDES := $(shell pkg-config --cflags OpenCL 2>/dev/null || echo "")

.PHONY: all clean detect_arch ptx-demos

all: detect_arch 01-vector-addition 02-ptx-assembly 03-gpu-programming-methods 04-gpu-architecture 05-neural-network 06-cnn-convolution 07-attention-mechanism 08-profiling-tracing 09-gpu-extension 10-cpu-gpu-profiling-boundaries 11-fine-grained-gpu-modifications 12-advanced-gpu-customizations 13-low-latency-gpu-packet-processing 14-cuda-function-annotations 15-opencl-vector-addition ptx-demos

# PTX Demo targets
ptx-demos: ptx_demo device_ptx_demo simple_ptx_demo

detect_arch:
	@echo "Detected GPU architecture: $(ARCH)"
	@if [ "$(ARCH)" = "sm_61" ]; then \
		echo "Using default sm_61 architecture. To use actual GPU architecture:"; \
		echo "1. Create detect_arch.cu with the code to detect architecture"; \
		echo "2. Or manually set architecture in Makefile"; \
	fi

01-vector-addition: 01-vector-addition.cu
	$(NVCC) $(NVCC_FLAGS) -o 01-vector-addition 01-vector-addition.cu

02-ptx-assembly: 02-ptx-assembly.cu
	$(NVCC) $(NVCC_PTX_FLAGS) -lcuda -o 02-ptx-assembly 02-ptx-assembly.cu

03-gpu-programming-methods: 03-gpu-programming-methods.cu
	$(NVCC) $(NVCC_DP_FLAGS) -o 03-gpu-programming-methods 03-gpu-programming-methods.cu

04-gpu-architecture: 04-gpu-architecture.cu
	$(NVCC) $(NVCC_FLAGS) -o 04-gpu-architecture 04-gpu-architecture.cu

05-neural-network: 05-neural-network.cu
	$(NVCC) $(NVCC_FLAGS) -o 05-neural-network 05-neural-network.cu

06-cnn-convolution: 06-cnn-convolution.cu
	$(NVCC) $(NVCC_FLAGS) -o 06-cnn-convolution 06-cnn-convolution.cu

07-attention-mechanism: 07-attention-mechanism.cu
	$(NVCC) $(NVCC_FLAGS) -o 07-attention-mechanism 07-attention-mechanism.cu

08-profiling-tracing: 08-profiling-tracing.cu
	$(NVCC) $(NVCC_PROF_FLAGS) -o 08-profiling-tracing 08-profiling-tracing.cu

09-gpu-extension: 09-gpu-extension.cu
	$(NVCC) $(NVCC_EXT_FLAGS) -o 09-gpu-extension 09-gpu-extension.cu

10-cpu-gpu-profiling-boundaries: 10-cpu-gpu-profiling-boundaries.cu
	$(NVCC) $(NVCC_PROF_FLAGS) -o 10-cpu-gpu-profiling-boundaries 10-cpu-gpu-profiling-boundaries.cu

11-fine-grained-gpu-modifications: 11-fine-grained-gpu-modifications.cu
	$(NVCC) $(NVCC_EXT_FLAGS) -o 11-fine-grained-gpu-modifications 11-fine-grained-gpu-modifications.cu

12-advanced-gpu-customizations: 12-advanced-gpu-customizations.cu
	$(NVCC) $(NVCC_EXT_FLAGS) -o 12-advanced-gpu-customizations 12-advanced-gpu-customizations.cu

13-low-latency-gpu-packet-processing: 13-low-latency-gpu-packet-processing.cu
	$(NVCC) $(NVCC_EXT_FLAGS) -o 13-low-latency-gpu-packet-processing 13-low-latency-gpu-packet-processing.cu

14-cuda-function-annotations: 14-cuda-function-annotations.cu
	$(NVCC) $(NVCC_FLAGS) -o 14-cuda-function-annotations 14-cuda-function-annotations.cu

# OpenCL example
15-opencl-vector-addition: 15-opencl-vector-addition.c
	@echo "Building OpenCL example..."
	@echo "Using OpenCL includes: $(OPENCL_INCLUDES)"
	@echo "Using OpenCL libs: $(OPENCL_LIBS)"
	$(CC) $(OPENCL_FLAGS) $(OPENCL_INCLUDES) -o 15-opencl-vector-addition 15-opencl-vector-addition.c $(OPENCL_LIBS) -lm

# PTX Demo builds
# Generate PTX file from CUDA kernel
vector_add.ptx: vector_add_kernel.cu
	$(NVCC) -ptx $< -o $@

# Main PTX loading demo
ptx_demo: 02-ptx-assembly.cu vector_add.ptx
	$(NVCC) $(NVCC_PTX_FLAGS) -lcuda $< -o $@

# Simple device PTX demo
simple_ptx_demo: simple-device-ptx.cu
	$(NVCC) $(NVCC_PTX_FLAGS) $< -o $@

clean:
	rm -f 01-vector-addition 02-ptx-assembly 03-gpu-programming-methods 04-gpu-architecture 05-neural-network 06-cnn-convolution 07-attention-mechanism 08-profiling-tracing 09-gpu-extension 10-cpu-gpu-profiling-boundaries 11-fine-grained-gpu-modifications 12-advanced-gpu-customizations 13-low-latency-gpu-packet-processing 14-cuda-function-annotations 15-opencl-vector-addition ptx_demo device_ptx_demo simple_ptx_demo vector_add.ptx 