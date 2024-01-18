# Building and run bpftime

## Table of Contents

- [Building and run bpftime](#building-and-run-bpftime)
  - [Table of Contents](#table-of-contents)
  - [Use docker image](#use-docker-image)
  - [Install Dependencies](#install-dependencies)
    - [Build and install cli tool](#build-and-install-cli-tool)
  - [Compilation for bpftime](#compilation-for-bpftime)
  - [Compile only the vm (No runtime, No uprobe)](#compile-only-the-vm-no-runtime-no-uprobe)
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
sudo apt-get update && apt-get install \
        libelf1 libelf-dev zlib1g-dev make cmake git libboost1.74-all-dev \
        binutils-dev libyaml-cpp-dev ca-certificates clang llvm
git submodule update --init --recursive
```

We've tested on Ubuntu 23.04. The recommended `gcc` >= 12.0.0 `clang` >= 16.0.0

On Ubuntu 20.04, you may need to manually switch to gcc-12.

### Build and install cli tool

```bash
make release && make install # Build and install the runtime
export PATH=$PATH:~/.bpftime
```

Then you can run cli:

```console
$ bpftime
Usage: bpftime [OPTIONS] <COMMAND>
...
```

## Compilation for bpftime

Build the complete runtime in release mode(With ubpf jit):

```bash
make release
```

Build the complete runtime in debug mode(With ubpf jit):

```bash
make debug
```

Build the complete runtime in release mode(With llvm jit):

```bash
make release-with-llvm-jit
```

## Compile only the vm (No runtime, No uprobe)

For a lightweight build without the runtime (only vm library and LLVM JIT):

```bash
make build-vm # build the simple vm with a simple jit
make build-llvm # build the vm with llvm jit
```

## Testing

Run the test suite for runtime to validate the implementation:

```bash
make unit-test
```
