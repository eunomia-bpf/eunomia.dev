NVCC = nvcc

# Auto-detect GPU architecture
ARCH := $(shell $(NVCC) -run ./detect_arch.cu 2>/dev/null || echo "sm_61")

NVCC_FLAGS = -O3 -arch=$(ARCH)
NVCC_DP_FLAGS = -O3 -arch=$(ARCH) -rdc=true
NVCC_PROF_FLAGS = -O3 -arch=$(ARCH) -lcupti -lnvToolsExt
NVCC_EXT_FLAGS = -O3 -arch=$(ARCH) -ldl -lpthread

.PHONY: all clean detect_arch

all: detect_arch 01-vector-addition 02-ptx-assembly 03-gpu-programming-methods 04-gpu-architecture 05-neural-network 06-cnn-convolution 07-attention-mechanism 08-profiling-tracing 09-gpu-extension

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
	$(NVCC) $(NVCC_FLAGS) -o 02-ptx-assembly 02-ptx-assembly.cu

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

clean:
	rm -f 01-vector-addition 02-ptx-assembly 03-gpu-programming-methods 04-gpu-architecture 05-neural-network 06-cnn-convolution 07-attention-mechanism 08-profiling-tracing 09-gpu-extension 