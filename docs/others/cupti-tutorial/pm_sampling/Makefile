#
# Copyright 2024 NVIDIA Corporation. All rights reserved
#
ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

CUDA_INSTALL_PATH ?= ../../../..
CUPTI_INSTALL_PATH ?= $(CUDA_INSTALL_PATH)/extras/CUPTI
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
INCLUDES := -I"$(CUDA_INSTALL_PATH)/include" -I$(CUPTI_INCLUDE_PATH) -I$(CUPTI_INSTALL_PATH)/include -I$(CUPTI_INSTALL_PATH)/samples/common

ifeq ($(OS),Windows_NT)
    LIB_PATH ?= $(CUPTI_INSTALL_PATH)\lib64
else
    EXTRAS_LIB_PATH := $(CUPTI_INSTALL_PATH)/lib64
    LIB_PATH ?= $(CUDA_INSTALL_PATH)/lib64
endif

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):$(LIB_PATH)
    LIBS= -lcuda -L $(LIB_PATH) -lcupti
    OBJ = obj
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(LIB_PATH)
        LIBS= -Xlinker -framework -Xlinker cuda -L $(LIB_PATH) -lcupti
    else
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
        LIBS = -L $(EXTRAS_LIB_PATH) -lcuda -L $(LIB_PATH) -lcupti
    endif
    OBJ = o
endif

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
            NVCCFLAGS += --qpp-config $(QPP_CONFIG_VERSION),gcc_ntoaarch64le -lsocket
        endif
    endif

    ifdef HOST_COMPILER
        NVCC_COMPILER := -ccbin $(HOST_COMPILER)
    endif
endif

# Gencode arguments
SMS ?= 75 80 86 87 89 90 100 101 120
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

pm_sampling: pm_sampling.$(OBJ)
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) -o $@ pm_sampling.$(OBJ) $(LIBS)

pm_sampling.$(OBJ): pm_sampling.cu
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) $(GENCODE_FLAGS) -lineinfo  -c $(INCLUDES) $<

run: pm_sampling
	./$<

clean:
	rm -f pm_sampling pm_sampling.$(OBJ)

