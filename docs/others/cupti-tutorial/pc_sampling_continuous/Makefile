#
# Copyright 2020-2021 NVIDIA Corporation. All rights reserved
#
ifndef OS
    OS   := $(shell uname)
    HOST_ARCH := $(shell uname -m)
endif

CUDA_INSTALL_PATH ?= ../../../..
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
INCLUDES := -I"$(CUDA_INSTALL_PATH)/include" -I$(CUPTI_INCLUDE_PATH) -I../../include -I../common

ifeq ($(OS),Windows_NT)
    LIB_PATH ?= ..\..\lib64
else
    EXTRAS_LIB_PATH := ../../lib64
    LIB_PATH ?= $(CUDA_INSTALL_PATH)/lib64
endif

NVCCFLAGS :=
ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):$(LIB_PATH)
    LIBS= -L $(LIB_PATH) -lcuda -lcupti -lpcsamplingutil -ldetours
    LIBNAME := pc_sampling_continuous.dll
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(LIB_PATH)
        LIBS= -Xlinker -framework -Xlinker cuda -L $(EXTRAS_LIB_PATH) -L $(LIB_PATH) -lcupti -lpcsamplingutil
    else
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
        LIBS = -L $(LIB_PATH) -lcuda -L $(EXTRAS_LIB_PATH) -lcupti -lpcsamplingutil
    endif
    LIBNAME := libpc_sampling_continuous.so
    NVCCFLAGS += -Xcompiler -fPIC
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

all: pc_sampling_continuous
pc_sampling_continuous: pc_sampling_continuous.cpp
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) $(INCLUDES) -o $(LIBNAME) -shared $< $(LIBS)
clean:
	rm -f $(LIBNAME)
