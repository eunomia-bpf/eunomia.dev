#
# Copyright 2011-2013 NVIDIA Corporation. All rights reserved
# 
INCLUDES=-I../../include

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

ifeq ($(OS),Windows_NT)
    LIB_PATH ?= ..\..\lib64
else
    LIB_PATH ?= ../../lib64
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
        LIBS= -lcuda -L $(LIB_PATH) -lcupti
    endif
    OBJ = o
endif

activity_trace: activity_trace.$(OBJ) vec.$(OBJ)
	nvcc -o $@ $^ $(LIBS)

activity_trace.$(OBJ): activity_trace.cpp
	nvcc -c $(INCLUDES) $<

vec.$(OBJ): vec.cu
	nvcc -c $(INCLUDES) $<

run: activity_trace
	./$<

clean:
	rm -f activity_trace activity_trace.$(OBJ) vec.$(OBJ)
