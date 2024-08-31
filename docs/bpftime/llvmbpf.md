# Userspace eBPF VM with LLVM JIT/AOT Compiler

[![Build and Test VM](https://github.com/eunomia-bpf/llvmbpf/actions/workflows/test-vm.yml/badge.svg)](https://github.com/eunomia-bpf/llvmbpf/actions/workflows/test-vm.yml)
[![codecov](https://codecov.io/gh/eunomia-bpf/llvmbpf/graph/badge.svg?token=ZQXHpOwDa1)](https://codecov.io/gh/eunomia-bpf/llvmbpf)

A high-performance, multi-architecture JIT/AOT compiler and virtual machine (VM) based on LLVM.

This component is part of the [bpftime](https://github.com/eunomia-bpf/bpftime) project but focuses solely on the core VM. It offers the following capabilities:

- Operates as `a standalone eBPF VM library` or compiler tool.
- Compiles eBPF bytecode into LLVM IR files.
- Compiles eBPF ELF files into AOTed native code ELF object files, which can be linked like C-compiled objects or loaded into llvmbpf.
- Loads and executes AOT-compiled ELF object files within the eBPF runtime.
- Supports eBPF helpers and maps lddw functions.

This library is optimized for performance, flexibility, and minimal dependencies. It does not include maps implement, helpers, verifiers, or loaders for eBPF applications, making it suitable as a lightweight, high-performance library.

For a comprehensive userspace eBPF runtime that includes support for maps, helpers, and seamless execution of Uprobe, syscall trace, XDP, and other eBPF programs—similar to kernel functionality but in userspace—please refer to the [bpftime](https://github.com/eunomia-bpf/bpftime) project.

- [Userspace eBPF VM with LLVM JIT/AOT Compiler](#userspace-ebpf-vm-with-llvm-jitaot-compiler)
  - [build project](#build-project)
  - [Usage](#usage)
    - [Use llvmbpf as a library](#use-llvmbpf-as-a-library)
    - [Use llvmbpf as a AOT compiler](#use-llvmbpf-as-a-aot-compiler)
    - [load eBPF bytecode from ELF file](#load-ebpf-bytecode-from-elf-file)
    - [Maps and data relocation support](#maps-and-data-relocation-support)
    - [Build into standalone binary for deployment](#build-into-standalone-binary-for-deployment)
  - [optimizaion](#optimizaion)
    - [inline the maps and helper function](#inline-the-maps-and-helper-function)
    - [Use original LLVM IR from C code](#use-original-llvm-ir-from-c-code)
  - [Test](#test)
    - [Unit test](#unit-test)
    - [Test with bpf-conformance](#test-with-bpf-conformance)
  - [License](#license)

## build project

```sh
sudo apt install llvm-15-dev libzstd-dev
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target all -j
```

## Usage

### Use llvmbpf as a library

See [example](example/main.cpp) of how to use the library as a vm:

```cpp
void run_ebpf_prog(const void *code, size_t code_len)
{
    uint64_t res = 0;
    llvmbpf_vm vm;

    res = vm.load_code(code, code_len);
    if (res) {
        return;
    }
    vm.register_external_function(2, "print", (void *)ffi_print_func);
    auto func = vm.compile();
    if (!func) {
        return;
    }
    int err = vm.exec(&bpf_mem, sizeof(bpf_mem), res);
    if (err != 0) {
        return;
    }
    printf("res = %" PRIu64 "\n", res);
}
```

### Use llvmbpf as a AOT compiler

Build with cli:

```sh
sudo apt-get install libelf1 libelf-dev
cmake -B build  -DBUILD_LLVM_AOT_CLI=1 
```

You can use the cli to generate the LLVM IR from eBPF bytecode:

```console
# ./build/cli/bpftime-vm build .github/assets/sum.bpf.o -emit-llvm > test.bpf.ll
# opt -O3 -S test.bpf.ll -opaque-pointers  -o test.opt.ll
# cat test.opt.ll
; ModuleID = 'test.bpf.ll'
source_filename = "bpf-jit"

; Function Attrs: nofree norecurse nosync nounwind memory(read, inaccessiblemem: none)
define i64 @bpf_main(ptr %0, i64 %1) local_unnamed_addr #0 {
setupBlock:
  %2 = ptrtoint ptr %0 to i64
  %3 = load i32, ptr %0, align 4
  %4 = icmp slt i32 %3, 1
  br i1 %4, label %bb_inst_30, label %bb_inst_15

bb_inst_15:                                       ; preds = %setupBlock, %bb_inst_15
  %storemerge32 = phi i32 [ %11, %bb_inst_15 ], [ 1, %setupBlock ]
  %stackBegin29.sroa.2.031 = phi i32 [ %10, %bb_inst_15 ], [ 0, %setupBlock ]
  %5 = sext i32 %storemerge32 to i64
  %6 = shl nsw i64 %5, 2
  %7 = add i64 %6, %2
  %8 = inttoptr i64 %7 to ptr
  %9 = load i32, ptr %8, align 4
  %10 = add i32 %9, %stackBegin29.sroa.2.031
  %11 = add i32 %storemerge32, 1
  %12 = icmp sgt i32 %11, %3
  br i1 %12, label %bb_inst_30, label %bb_inst_15

bb_inst_30:                                       ; preds = %bb_inst_15, %setupBlock
  %stackBegin29.sroa.2.0.lcssa = phi i32 [ 0, %setupBlock ], [ %10, %bb_inst_15 ]
  %13 = zext i32 %stackBegin29.sroa.2.0.lcssa to i64
  ret i64 %13
}

attributes #0 = { nofree norecurse nosync nounwind memory(read, inaccessiblemem: none) }
```

AOT Compile a eBPF program:

```console
# ./build/cli/bpftime-vm build .github/assets/sum.bpf.o
[2024-08-10 14:54:06.453] [info] [main.cpp:56] Processing program test
[2024-08-10 14:54:06.479] [info] [main.cpp:69] Program test written to ./test.o
```

Load and run a AOTed eBPF program:

```console
# echo "AwAAAAEAAAACAAAAAwAAAA==" | base64 -d > test.bin
# ./build/cli/bpftime-vm run test.o test.bin
[2024-08-10 14:57:16.986] [info] [llvm_jit_context.cpp:392] LLVM-JIT: Loading aot object
[2024-08-10 14:57:16.991] [info] [main.cpp:136] Program executed successfully. Return value: 6
```

See [Build into standalone binary for deployment](#build-into-standalone-binary-for-deployment) for more details.

### load eBPF bytecode from ELF file

You can use llvmbpf together with libbpf to load the eBPF bytecode directly from `bpf.o` ELF file. For example:

```c
  bpf_object *obj = bpf_object__open(ebpf_elf.c_str());
  if (!obj) {
    return 1;
  }
  std::unique_ptr<bpf_object, decltype(&bpf_object__close)> elf(
    obj, bpf_object__close);

  bpf_program *prog;
  for ((prog) = bpf_object__next_program((elf.get()), __null);
       (prog) != __null;
       (prog) = bpf_object__next_program((elf.get()), (prog))) {
    const char *name = bpf_program__name(prog);
    llvmbpf_vm vm;

    vm.load_code((const void *)bpf_program__insns(prog),
         (uint32_t)bpf_program__insn_cnt(prog) * 8);
  ...
  }
```

For complete code example, please refer to [cli](cli).

However, the `bpf.o` ELF file has no map and data relocation support. We would recommend using the bpftime to load and relocation the eBPF bytecode from ELF file. This include:

- Write a loader like normal kernel eBPF loader to load the eBPF bytecode, you can find a example [here](https://github.com/eunomia-bpf/bpftime/blob/master/example/xdp-counter/xdp-counter.c).
- The loader will use the libbpf, which support:
  - Relocation for map. The map id will be allocated by the loader and bpftime, you can use the map id to access map through the helpers.
  - The data can be accessed through the lddw helper function.
- After the loader load the eBPF bytecode and complete the relocation, you can use the [bpftimetool](https://eunomia.dev/zh/bpftime/documents/bpftimetool/) to dump the map information and eBPF bytecode.

### Maps and data relocation support

bpftime already has maps and data relocation support. The easiest way to use it is just use bpftime and write the loader and eBPF program like kernel eBPF. The `llvmbpf` libray provide a approach to interact with the maps.

See [example/maps.cpp](example/maps.cpp) of how to use the library as a vm and works with maps:

The eBPF can work with maps in two ways:

- Using helper functions to access the maps, like `bpf_map_lookup_elem`, `bpf_map_update_elem`, etc.
- Using maps as global variables in the eBPF program, and access the maps directly.

For a eBPF program like [https://github.com/eunomia-bpf/bpftime/blob/master/example/xdp-counter/](https://github.com/eunomia-bpf/bpftime/blob/master/example/xdp-counter/):

```c
// use map type define
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __type(key, __u32);
  __type(value, __u32);
  __uint(max_entries, CTRL_ARRAY_SIZE);
} ctl_array SEC(".maps");

// use global variable define
__u64 cntrs_array[CNTRS_ARRAY_SIZE];

SEC("xdp")
int xdp_pass(struct xdp_md* ctx) {
  void* data_end = (void*)(long)ctx->data_end;
  void* data = (void*)(long)ctx->data;
  __u32 ctl_flag_pos = 0;
  __u32 cntr_pos = 0;

  // access maps with helpers
  __u32* flag = bpf_map_lookup_elem(&ctl_array, &ctl_flag_pos);
  if (!flag || (*flag != 0)) {
    return XDP_PASS;
  };

  // access maps with global variables
  cntrs_array[cntr_pos]++;

  if (data + sizeof(struct ethhdr) > data_end)
    return XDP_DROP;
  swap_src_dst_mac(data);
  return XDP_TX;
}
```

We can define the map and access them like:

```cpp
uint32_t ctl_array[2] = { 0, 0 };
uint64_t cntrs_array[2] = { 0, 0 };

void *bpf_map_lookup_elem(uint64_t map_fd, void *key)
{
  std::cout << "bpf_map_lookup_elem " << map_fd << std::endl;
  if (map_fd == 5) {
    return &ctl_array[*(uint32_t *)key];
  } else if (map_fd == 6) {
    return &cntrs_array[*(uint32_t *)key];
  } else {
    return nullptr;
  }
  return 0;
}

uint64_t map_by_fd(uint32_t fd)
{
  std::cout << "map_by_fd " << fd << std::endl;
  return fd;
}

uint64_t map_val(uint64_t val)
{
  std::cout << "map_val " << val << std::endl;
  if (val == 5) {
    return (uint64_t)(void *)ctl_array;
  } else if (val == 6) {
    return (uint64_t)(void *)cntrs_array;
  } else {
    return 0;
  }
}

int main(int argc, char *argv[])
{
  auto code = xdp_counter_bytecode;
  size_t code_len = sizeof(xdp_counter_bytecode) - 1;
  uint64_t res = 0;
  llvmbpf_vm vm;

  res = vm.load_code(code, code_len);
  if (res) {
    std::cout << vm.get_error_message() << std::endl;
    exit(1);
  }
  vm.register_external_function(1, "bpf_map_lookup_elem",
              (void *)bpf_map_lookup_elem);
  // set the lddw helpers for accessing maps
  vm.set_lddw_helpers(map_by_fd, nullptr, map_val, nullptr, nullptr);
  auto func = vm.compile();
  if (!func) {
    std::cout << vm.get_error_message() << std::endl;
    exit(1);
  }
  // Map value (counter) should be 0
  std::cout << "cntrs_array[0] = " << cntrs_array[0] << std::endl;
  int err = vm.exec(&bpf_mem, sizeof(bpf_mem), res);
  std::cout << "\nreturn value = " << res << std::endl;
  // counter should be 1
  std::cout << "cntrs_array[0] = " << cntrs_array[0] << std::endl;
  ....
}
```

Reference:

- <https://prototype-kernel.readthedocs.io/en/latest/bpf/ebpf_maps.html>
- <https://www.ietf.org/archive/id/draft-ietf-bpf-isa-00.html#name-64-bit-immediate-instructio>

### Build into standalone binary for deployment

You can build the eBPF program into a standalone binary, which does not rely on any external libraries, and can be exec like nomal c code with helper and maps support.

This can help:

- Easily deploy the eBPF program to any machine without the need to install any dependencies.
- Avoid the overhead of loading the eBPF bytecode and maps at runtime.
- Suitable for microcontroller or embedded systems, which does not have a OS.

Take [https://github.com/eunomia-bpf/bpftime/blob/master/example/xdp-counter/](https://github.com/eunomia-bpf/bpftime/blob/master/example/xdp-counter/) as an example:

In the bpftime project:

```sh
# load the eBPF program with bpftime
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/xdp-counter/xdp-counter example/xdp-counter/.output/xdp-counter.bpf.o veth1
# dump the map and eBPF bytecode define
./build/tools/bpftimetool/bpftimetool export res.json
# build the eBPF program into llvm IR
./build/tools/aot/bpftime-aot compile --emit_llvm 1>xdp-counter.ll
```

You can see [example/xdp-counter.json](example/xdp-counter.json) for an example json file dump by bpftime.

The result xdp-counter.ll can be found in [example/standalone/xdp-counter.ll](example/standalone/xdp-counter.ll).

Then you can write a C code and compile it with the llvm IR:

```c
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

int bpf_main(void* ctx, uint64_t size);

uint32_t ctl_array[2] = { 0, 0 };
uint64_t cntrs_array[2] = { 0, 0 };

void *_bpf_helper_ext_0001(uint64_t map_fd, void *key)
{
  printf("bpf_map_lookup_elem %lu\n", map_fd);
  if (map_fd == 5) {
    return &ctl_array[*(uint32_t *)key];
  } else if (map_fd == 6) {
    return &cntrs_array[*(uint32_t *)key];
  } else {
    return NULL;
  }
  return 0;
}

void* __lddw_helper_map_val(uint64_t val)
{
    printf("map_val %lu\n", val);
    if (val == 5) {
        return (void *)ctl_array;
    } else if (val == 6) {
        return (void *)cntrs_array;
    } else {
        return NULL;
    }
}

uint8_t bpf_mem[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88 };

int main() {
    printf("The value of cntrs_array[0] is %" PRIu64 "\n", cntrs_array[0]);
    printf("calling ebpf program...\n");
    bpf_main(bpf_mem, sizeof(bpf_mem));
    printf("The value of cntrs_array[0] is %" PRIu64 "\n", cntrs_array[0]);
    printf("calling ebpf program...\n");
    bpf_main(bpf_mem, sizeof(bpf_mem));
    printf("The value of cntrs_array[0] is %" PRIu64 "\n", cntrs_array[0]);
    return 0;
}
```

Compile the C code with the llvm IR:

```sh
clang -g main.c xdp-counter.ll -o standalone 
```

And you can run the `standalone` eBPF program directly.

## optimizaion

Based on the AOT compiler, we can apply some optimization strategies:

### inline the maps and helper function

Inline the maps and helper function into the eBPF program, so that the eBPF program can be optimized with `const propagation`, `dead code elimination`, etc by the LLVM optimizer. llvmbpf can also eliminate the cost of function calls.

Prepare a C code:

```c

uint32_t ctl_array[2] = { 0, 0 };
uint64_t cntrs_array[2] = { 0, 0 };

void *_bpf_helper_ext_0001(uint64_t map_fd, void *key)
{
  if (map_fd == 5) {
    return &ctl_array[*(uint32_t *)key];
  } else if (map_fd == 6) {
    return &cntrs_array[*(uint32_t *)key];
  } else {
    return NULL;
  }
  return 0;
}

void* __lddw_helper_map_val(uint64_t val)
{
    if (val == 5) {
        return (void *)ctl_array;
    } else if (val == 6) {
        return (void *)cntrs_array;
    } else {
        return NULL;
    }
}
```

Merge the modules with `llvm-link` and inline them:

```sh
clang -S -O3 -emit-llvm libmap.c -o libmap.ll
llvm-link -S -o xdp-counter-inline.ll xdp-counter.ll libmap.ll
opt --always-inline -S xdp-counter-inline.ll -o xdp-counter-inline.ll
clang -O3 -g -c xdp-counter-inline.ll -o inline.o
```

Run the code with cli:

```c
./build/cli/bpftime-vm run example/inline/inline.o test.bin
```

Or you can compile as standalone binary and link with the C code:

```console
$ clang -O3 example/inline/inline.o example/inline/main.c -o inline
$ /workspaces/llvmbpf/inline
calling ebpf program...
return value = 1
```

### Use original LLVM IR from C code

eBPF is a instruction set define for verification, but may not be the best for performance.

llvmbpf also support using the original LLVM IR from C code. See [example/load-llvm-ir](example/load-llvm-ir) for an example. You can:

- Compile the C code to eBPF for verify
- Compile the C code to LLVM IR and native code for execution in the VM.

The C code:

```c
int _bpf_helper_ext_0006(const char *fmt, ... );

int bpf_main(void* ctx, int size) {
    _bpf_helper_ext_0006("hello world: %d\n", size);
    return 0;
}
```

You can compile it with `clang -g -c bpf_module.c -o bpf_module.o`, and Run the code with cli:

```c
./build/cli/bpftime-vm run example/load-llvm-ir/bpf_module.o test.bin
```

## Test

### Unit test

Compile:

```sh
sudo apt install llvm-15-dev libzstd-dev
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBPFTIME_ENABLE_UNIT_TESTING=1 -DBPFTIME_ENABLE_CODE_COVERAGE=1
cmake --build build --target all -j
```

The unit tests can be found at `build/test/unit-test/llvm_jit_tests`.

### Test with bpf-conformance

See the CI in [.github/workflows/bpf_conformance.yml](.github/workflows/bpf_conformance.yml) for how to run the bpf-conformance tests.

The test result can be found in <https://eunomia-bpf.github.io/llvmbpf/bpf_conformance_results.txt>

## License

MIT
