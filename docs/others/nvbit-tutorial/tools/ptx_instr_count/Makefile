# Makefile for PTX POC - NVBit tool with hand-written PTX device functions
#
# Build pipeline:
#   inject_funcs.ptx -> ptxas -astoolspatch -> cubin -> fatbinary
#   inject_funcs_embed.S (.incbin fatbin) -> gcc -> inject_funcs_embed.o (.nv_fatbin section)
#   inject_funcs_reg.cpp -> g++ -> inject_funcs_reg.o (.nvFatBinSegment wrapper)
#   instr_count.cu -> nvcc -> instr_count.o
#   link all -> ptx_instr_count.so

NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas
FATBINARY=fatbinary

NVBIT_PATH=../../core
INCLUDES=-I$(NVBIT_PATH)
LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

# Detect GPU architecture for cubin compilation
# Default to sm_120 (RTX 5090), but also include PTX for JIT fallback
GPU_ARCH?=120

NVBIT_TOOL=ptx_instr_count.so

all: $(NVBIT_TOOL)

# === Step 1: Compile hand-written PTX to cubin with -astoolspatch ===
inject_funcs.sm_$(GPU_ARCH).cubin: inject_funcs.ptx
	@echo "  [PTX->CUBIN] Compiling hand-written PTX for sm_$(GPU_ARCH)..."
	$(PTXAS) -astoolspatch -arch=sm_$(GPU_ARCH) $< -o $@

# === Step 2: Package cubin + PTX into fatbinary ===
inject_funcs.fatbin: inject_funcs.sm_$(GPU_ARCH).cubin inject_funcs.ptx
	@echo "  [FATBIN] Creating fat binary (cubin + PTX JIT fallback)..."
	$(FATBINARY) --create=$@ -64 \
		--image=profile=sm_$(GPU_ARCH),file=inject_funcs.sm_$(GPU_ARCH).cubin \
		--image=profile=compute_70,file=inject_funcs.ptx

# === Step 3: Embed fatbin into .nv_fatbin ELF section via assembly ===
# Uses .incbin to embed the fatbin with hidden visibility, ensuring
# R_X86_64_RELATIVE relocation that cuobjdump can resolve statically.
inject_funcs_embed.o: inject_funcs_embed.S inject_funcs.fatbin
	@echo "  [ASM] Embedding fatbin into .nv_fatbin ELF section..."
	gcc -c -fPIC $< -o $@

# === Step 4: Compile host registration stub ===
inject_funcs_reg.o: inject_funcs_reg.cpp
	@echo "  [CXX] Compiling registration stub..."
	g++ -fPIC -c $< -o $@

# === Step 5: Compile host-side NVBit tool code (still uses nvcc for .cu) ===
instr_count.o: instr_count.cu
	$(NVCC) -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -O3 -Xcompiler -fPIC $< -o $@

# === Step 6: Link everything into the final .so ===
$(NVBIT_TOOL): instr_count.o inject_funcs_reg.o inject_funcs_embed.o $(NVBIT_PATH)/libnvbit.a
	@echo "  [LINK] Building $(NVBIT_TOOL)..."
	g++ -shared -fPIC $^ $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -lpthread -ldl -o $@
	@echo "  [DONE] Built $(NVBIT_TOOL) with hand-written PTX device functions!"

clean:
	rm -f *.so *.o *.cubin *.fatbin

.PHONY: all clean
