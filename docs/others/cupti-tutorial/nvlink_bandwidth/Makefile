#
# Copyright 2015-2021 NVIDIA Corporation. All rights reserved.
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

ifeq ($(OS),Windows_NT)
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

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):$(LIB_PATH)
    LIBS= -lcuda -L $(LIB_PATH) -lcupti
    OBJ = obj
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(LIB_PATH)
        LIBS= -Xlinker -framework -Xlinker cuda -L $(LIB_PATH) -lcupti
    else
        LIBS :=
        ifeq ($(HOST_ARCH), $(TARGET_ARCH))
            export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
            LIBS = -L $(EXTRAS_LIB_PATH)
        endif
        LIBS += $(TARGET_CUDA_PATH) -lcuda -L $(LIB_PATH) -lcupti
    endif
    OBJ = o
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
        NVCC_COMPILER := -ccbin $(HOST_COMPILER)
    endif
endif

# Gencode arguments
SMS ?= 60 70
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

nvlink_bandwidth: nvlink_bandwidth.$(OBJ)
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) -o $@ nvlink_bandwidth.$(OBJ) $(LIBS) $(INCLUDES)

nvlink_bandwidth.$(OBJ): nvlink_bandwidth.cu
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) $(GENCODE_FLAGS) -c $(INCLUDES) $<

run: nvlink_bandwidth
	./$< --cpu-to-gpu

clean:
	rm -f nvlink_bandwidth nvlink_bandwidth.$(OBJ)

