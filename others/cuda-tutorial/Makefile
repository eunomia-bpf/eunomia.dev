NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_61
NVCC_DP_FLAGS = -O3 -arch=sm_61 -rdc=true

all: basic01 basic02 basic03 basic04 basic05 basic06 basic07

basic01: basic01.cu
	$(NVCC) $(NVCC_FLAGS) -o basic01 basic01.cu

basic02: basic02.cu
	$(NVCC) $(NVCC_FLAGS) -o basic02 basic02.cu

basic03: basic03.cu
	$(NVCC) $(NVCC_DP_FLAGS) -o basic03 basic03.cu

basic04: basic04.cu
	$(NVCC) $(NVCC_FLAGS) -o basic04 basic04.cu

basic05: basic05.cu
	$(NVCC) $(NVCC_FLAGS) -o basic05 basic05.cu

basic06: basic06.cu
	$(NVCC) $(NVCC_FLAGS) -o basic06 basic06.cu

basic07: basic07.cu
	$(NVCC) $(NVCC_FLAGS) -o basic07 basic07.cu

clean:
	rm -f basic01 basic02 basic03 basic04 basic05 basic06 basic07 