#
# Copyright 2021 NVIDIA Corporation. All rights reserved
#
ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

CUDA_INSTALL_PATH ?= ../../../..
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
INCLUDES := -I"$(CUDA_INSTALL_PATH)/include" -I$(CUPTI_INCLUDE_PATH) -I../../include -I../common

TARGET_ARCH ?= $(HOST_ARCH)
TARGET_OS ?= $(shell uname | tr A-Z a-z)

# Set required library paths.
# In the case of cross-compilation, set the libs to the correct ones under /usr/local/cuda/targets/<TARGET_ARCH>-<TARGET_OS>/lib

ifeq ($(OS), Windows_NT)
    LIB_PATH ?= ..\..\lib64
else
    ifneq ($(TARGET_ARCH), $(HOST_ARCH))
        INCLUDES += -I$(CUDA_INSTALL_PATH)/targets/$(HOST_ARCH)-$(shell uname | tr A-Z a-z)/include
        INCLUDES += -I$(CUDA_INSTALL_PATH)/targets/$(TARGET_ARCH)-$(TARGET_OS)/include
        LIB_PATH ?= $(CUDA_INSTALL_PATH)/targets/$(TARGET_ARCH)-$(TARGET_OS)/lib
        TARGET_CUDA_PATH = -L $(LIB_PATH)/stubs
    else
        EXTRAS_LIB_PATH := ../../lib64
        LIB_PATH ?= $(CUDA_INSTALL_PATH)/lib64
    endif
endif

ifeq ($(OS), Windows_NT)
    LIBS = -lcuda -L $(LIB_PATH) -lcheckpoint -lcupti
    OBJ = obj
    LIBEXT = lib
    LIBPREFIX =
    BINEXT = .exe
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(LIB_PATH)
        LIBS = -Xlinker -framework -Xlinker cuda -L $(LIB_PATH) -lcheckpoint -lcupti
    else
        LIBS :=
        ifeq ($(HOST_ARCH), $(TARGET_ARCH))
            export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
            LIBS = -L $(EXTRAS_LIB_PATH)
        endif
        LIBS += $(TARGET_CUDA_PATH) -lcuda -L $(LIB_PATH) -lcheckpoint -lcupti
    endif
    OBJ = o
    LIBEXT = a
    LIBPREFIX = lib
    BINEXT =
endif

# Point to the necessary cross-compiler.
NVCCFLAGS :=
ifneq ($(TARGET_ARCH), $(HOST_ARCH))
    ifeq ($(TARGET_ARCH), aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/q++
            ifndef QPP_CONFIG_VERSION
                QPP_CONFIG_VERSION = 12.2.0
            endif
            $(info QPP_CONFIG_VERSION = $(QPP_CONFIG_VERSION))
            NVCCFLAGS := --qpp-config $(QPP_CONFIG_VERSION),gcc_ntoaarch64le -lsocket
        endif
    endif

    ifdef HOST_COMPILER
        NVCC_COMPILER = -ccbin $(HOST_COMPILER)
    endif
endif

# Gencode arguments
SMS ?= 50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 101 120
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

checkpoint_kernels: checkpoint_kernels.$(OBJ)
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) -o $@ $^ $(LIBS) $(INCLUDES)

checkpoint_kernels.$(OBJ): checkpoint_kernels.cu
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) $(GENCODE_FLAGS) -c $(INCLUDES) $<

run: checkpoint_kernels
	./$<

clean:
ifeq ($(OS), Windows_NT)
	del checkpoint_kernels.exe checkpoint_kernels.lib checkpoint_kernels.exp
else
	rm -f checkpoint_kernels
endif
