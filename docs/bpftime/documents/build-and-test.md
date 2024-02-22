# Building and Test

## Table of Contents

- [Building and Test](#building-and-test)
  - [Table of Contents](#table-of-contents)
  - [Use docker image](#use-docker-image)
  - [Install Dependencies](#install-dependencies)
    - [Build and install cli tool](#build-and-install-cli-tool)
  - [Compilation for bpftime](#compilation-for-bpftime)
  - [Compile only the vm (No runtime, No uprobe)](#compile-only-the-vm-no-runtime-no-uprobe)
  - [Compile with LTO enabled](#compile-with-lto-enabled)
  - [Compile with userspace verifier](#compile-with-userspace-verifier)
  - [More compile options](#more-compile-options)
  - [Testing](#testing)

## Use docker image

We provide a docker image for building and testing bpftime.

```bash
docker pull ghcr.io/eunomia-bpf/bpftime:latest
docker run -it --rm -v "$(pwd)":/workdir -w /workdir ghcr.io/eunomia-bpf/bpftime:latest /bin/bash
```

Or build the docker from dockerfile:

```bash
git submodule update --init --recursive
docker build .
```

## Install Dependencies

Install the required packages:

```bash
sudo apt-get update && sudo apt-get install \
        libelf1 libelf-dev zlib1g-dev make cmake git libboost1.74-all-dev \
        binutils-dev libyaml-cpp-dev ca-certificates clang llvm pkg-config
git submodule update --init --recursive
```

We've tested on Ubuntu 23.04. The recommended `gcc` >= 12.0.0 `clang` >= 16.0.0

On Ubuntu 20.04, you may need to manually switch to gcc-12.

### Build and install all things

Install all things that could be installed to `~/.bpftime`, includes:
- `bpftime`: A cli tool used for injecting agent & server to userspace programs
- `bpftime-vm`: A cli tool used for compiling eBPF programs into native programs, or run the compiled program
- `bpftimetool`: A cli tool used to manage things stored in shared memory, such as the data of maps or programs
- `bpftime_daemon`: An executable used for implementing the similar thing like syscall server, but don't need to be injected to the userspace program
- `libbpftime-agent.so`, `libbpftime-agent-transformer.so`: Libraries needed by bpftime agent
- `libbpftime-syscall-server.so`: Library needed by bpftime syscall server

Build with makefile:

```bash
make release JOBS=$(nproc) # Build and install the runtime
export PATH=$PATH:~/.bpftime
```

Or you can also build with `cmake`(The Makefile is a wrapper of cmake commands):

```bash
cmake -Bbuild  -DCMAKE_BUILD_TYPE:STRING=Release \
           -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO
cmake --build build --config Release --target install
export PATH=$PATH:~/.bpftime
```

Then you can run cli:

```console
$ bpftime
Usage: bpftime [OPTIONS] <COMMAND>
...
```

See the [Makefile](https://github.com/eunomia-bpf/bpftime/blob/master/Makefile) for some common commands.

## Detailed things about building

We use cmake as build system. You may be interested in the following cmake options:

- `CMAKE_BUILD_TYPE`: Specify the build type. It could be `Debug`, `Release`, `MinSizeRel` or `RelWithDebInfo`. If you are not going to debug bpftime, you just need to set it to `Release`. Default to `Debug`.
- `BPFTIME_ENABLE_UNIT_TESTING`: Whether to build unit test targets. See `Testing targets` for details. Default to `NO`.
- `BPFTIME_ENABLE_LTO`: Whether to enable Link Time Optimization. Enabling this may increase the compile time, but it may lead to a better performance. Default to `No`.
- `BPFTIME_LLVM_JIT`: Whether to use LLVM JIT as the ebpf runtime. Requires LLVM >= 15. It's recommended to enable this, since the ubpf intepreter is no longer maintained. Default to `NO`.
- `LLVM_DIR`: Specify the installing directory of LLVM. CMake may not discover the LLVM installation by default. Set this option to the directory that contains `LLVMConfig.cmake`, such as `/usr/lib/llvm-15/cmake` on Ubuntu

### Build and install the complete runtime in release mode(With ubpf jit):

```bash
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DBPFTIME_ENABLE_LTO=NO
cmake --build build --config Release --target install
```

### Build and install the complete runtime in debug mode(With ubpf jit):

```bash
cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug -DBPFTIME_ENABLE_LTO=NO
cmake --build build --config Debug --target install
```

### Build and install the complete runtime in release mode(With llvm jit):

```bash
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DBPFTIME_ENABLE_LTO=NO -DBPFTIME_LLVM_JIT=YES
cmake --build build --config RelWithDebInfo --target install
```

### Compile with LTO enabled

Just set `BPFTIME_ENABLE_LTO` to `YES`

For example, build  the package, with llvm-jit and LTO enabled:

```sh
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DBPFTIME_ENABLE_LTO=YES -DBPFTIME_LLVM_JIT=YES
cmake --build build --config RelWithDebInfo --target install
```

### Compile with userspace verifier 

Note that we are using https://github.com/vbpf/ebpf-verifier as userspace verifier. It's not perfect, and may not support some features (such as ringbuf)

```sh
cmake -DBPFTIME_LLVM_JIT=NO -DENABLE_EBPF_VERIFIER=YES -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build --config Release --target install
```

### Testing targets

We have some targets for unit testing, they are:
- `bpftime_daemon_tests`
- `bpftime_runtime_tests`
- `llvm_jit_tests`

These targets will only be enabled when `BPFTIME_ENABLE_UNIT_TESTING` was set to `YES`.

Build and run them to test, for example:
```sh
cmake -DBPFTIME_LLVM_JIT=YES -DBPFTIME_ENABLE_UNIT_TESTING=YES -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build --config RelWithDebInfo --target bpftime_runtime_tests
sudo ./build/runtime/unit-test/bpftime_runtime_tests
```

## Compile only the vm (No runtime, No uprobe)

For a lightweight build without the runtime (only vm library and LLVM JIT):

```bash
make build-vm # build the simple vm with a simple jit
make build-llvm # build the vm with llvm jit
```

## More compile options

See <https://github.com/eunomia-bpf/bpftime/blob/master/cmake/StandardSettings.cmake> for all cmake build options.

