#!/bin/bash
#
# Installation script for CUPTI samples
#

set -e  # Exit on error

echo "CUPTI Samples Installation Script"
echo "=================================="

# Check if CUDA is installed
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    else
        echo "CUDA_HOME environment variable is not set and CUDA is not found in /usr/local/cuda"
        echo "Please install CUDA or set CUDA_HOME environment variable"
        exit 1
    fi
fi

echo "Using CUDA installation at: $CUDA_HOME"

# Create lib64 directory if it doesn't exist
if [ ! -d "lib64" ]; then
    mkdir -p lib64
    echo "Created lib64 directory"
fi

# Link CUPTI libraries
echo "Creating symlinks to CUPTI libraries..."
ln -sf $CUDA_HOME/lib64/libcupti.so* lib64/
ln -sf $CUDA_HOME/lib64/libnvperf_host.so* lib64/
ln -sf $CUDA_HOME/lib64/libnvperf_target.so* lib64/

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

echo "Detected CUDA version: $CUDA_VERSION (Major: $CUDA_MAJOR, Minor: $CUDA_MINOR)"

# Ensure include directories exist
mkdir -p extensions/include/profilerhost_util
mkdir -p extensions/include/c_util

# Copy the ScopeExit.h header file if it doesn't exist
if [ ! -f "extensions/include/c_util/ScopeExit.h" ]; then
    echo "Creating ScopeExit.h header..."
    cat > extensions/include/c_util/ScopeExit.h << 'EOF'
#pragma once
#include <utility>

#define NV_ANONYMOUS_VARIABLE_DIRECT(name, line) name ## line
#define NV_ANONYMOUS_VARIABLE_INDIRECT(name, line) NV_ANONYMOUS_VARIABLE_DIRECT(name, line)
#define NV_ANONYMOUS_VARIABLE(name) NV_ANONYMOUS_VARIABLE_INDIRECT(name, __LINE__)

namespace detail {
    template <typename F>
    class ScopeExit {
        F func;
        bool active;
    public:
        ScopeExit(F&& f) : func(std::move(f)), active(true) {}
        ~ScopeExit() { if (active) func(); }
        void release() { active = false; }
        ScopeExit(ScopeExit&& rhs) : func(std::move(rhs.func)), active(rhs.active) { rhs.release(); }
        ScopeExit(const ScopeExit&) = delete;
        void operator=(const ScopeExit&) = delete;
    };
    template <typename F>
    ScopeExit<F> MakeScopeExit(F&& f) { return ScopeExit<F>(std::forward<F>(f)); }
    template <typename F>
    ScopeExit<F> MoveScopeExit(F& f) { return ScopeExit<F>(std::move(f)); }
}

#define SCOPE_EXIT(func) const auto NV_ANONYMOUS_VARIABLE_INDIRECT(EXIT, __LINE__) = detail::MoveScopeExit([&](){func;})
EOF
fi

# Build the profilerHostUtil library
echo "Building profilerHostUtil library..."
cd extensions && make && cd ..

# Copy the library to lib64
cp extensions/src/profilerhost_util/libprofilerHostUtil.* lib64/

echo "Setting up environment variables..."
# Export library path for running samples
export LD_LIBRARY_PATH=$(pwd)/lib64:$LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=$(pwd)/lib64:\$LD_LIBRARY_PATH" > setup_env.sh

echo "Installation completed successfully!"
echo "To set up the environment for running samples, run:"
echo "source setup_env.sh"
echo ""
echo "Using actual implementation of profilerHostUtil compatible with CUDA $CUDA_VERSION." 