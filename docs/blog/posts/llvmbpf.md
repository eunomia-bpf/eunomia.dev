---
date: 2024-09-10
---

# Building High-Performance Userspace eBPF VMs with LLVM

We are excited to introduce [**llvmbpf**](https://github.com/eunomia-bpf/llvmbpf), a new project aimed at empowering developers with a high-performance, multi-architecture eBPF virtual machine (VM) that leverages the LLVM framework for Just-In-Time (JIT) and Ahead-Of-Time (AOT) compilation.

This component is part of the [bpftime](https://github.com/eunomia-bpf/bpftime) project but focuses solely on the core VM. It operates as a standalone eBPF VM library or a compiler tool. This library is optimized for performance, flexibility, and minimal dependencies, making it easy to integrate into various environments without unnecessary overhead.
<!-- more -->

## Why llvmbpf?

Although there are several userspace eBPF runtimes available, we built llvmbpf to address specific needs that existing solutions may not fully satisfy:

1. **AOT Compiler**: The ability to compile eBPF bytecode into native ELF object files allows developers to deploy pre-compiled eBPF programs, ensuring high performance and efficiency, especially in resource-constrained environments. Additionally, it can allow you to experiment with different optimization techniques based on LLVM IR, providing more flexibility and control over the compilation process.

2. **Standalone Deployment**: With llvmbpf, you can build eBPF programs into standalone binaries that don’t require external dependencies. This feature is particularly useful for deploying eBPF programs on embedded systems, microcontrollers, or other environments where installing additional software is impractical. Compared to native C code development, this ensures the eBPF part is verified after integration with the verifier.

3. **All-Architecture Support**: llvmbpf is designed to be compatible across multiple architectures, making it versatile for a wide range of hardware platforms.

4. **Maps and Relocation Support**: Unlike many other userspace eBPF solutions, llvmbpf provides robust support for maps, data relocation, and `lddw` helper functions, allowing for the creation of more complex and powerful eBPF programs.

5. **Extensible Optimization Approaches**: Leveraging LLVM’s powerful optimization capabilities, llvmbpf allows for advanced optimizations such as inlining maps and helper functions, as well as using original LLVM IR for enhanced performance. 

In this blog, we’ll walk through some practical examples of how to use llvmbpf, highlighting its core features and capabilities.

For a comprehensive userspace eBPF runtime that includes support for maps, helpers, and seamless execution of Uprobe, syscall trace, XDP, and other eBPF programs—similar to kernel functionality but in userspace—please refer to the [bpftime](https://github.com/eunomia-bpf/bpftime) project.

## Getting Started with llvmbpf

### Using llvmbpf as a Library

llvmbpf can be used as a library within your application to load and execute eBPF programs. Here’s a basic example:

```cpp
void run_ebpf_prog(const void *code, size_t code_len) {
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

This snippet shows how you can load eBPF bytecode, register external functions, and execute the program within the VM.

### Using llvmbpf as an AOT Compiler

One of the most powerful features of llvmbpf is its ability to function as an AOT compiler, converting eBPF bytecode into native ELF object files. This approach not only boosts performance but also simplifies the deployment of eBPF programs.

You can use the CLI to generate LLVM IR from eBPF bytecode:

```console
# ./build/cli/bpftime-vm build .github/assets/sum.bpf.o -emit-llvm > test.bpf.ll
# opt -O3 -S test.bpf.ll -opaque-pointers -o test.opt.ll
# cat test.opt.ll
```

AOT Compile an eBPF program:

```console
# ./build/cli/bpftime-vm build .github/assets/sum.bpf.o
[info] Processing program test
[info] Program test written to ./test.o
```

Load and run an AOT-compiled eBPF program:

```console
# echo "AwAAAAEAAAACAAAAAwAAAA==" | base64 -d > test.bin
# ./build/cli/bpftime-vm run test.o test.bin
[info] LLVM-JIT: Loading aot object
[info] Program executed successfully. Return value: 6
```

The resulting ELF object file can be linked with other object files or loaded directly into the llvmbpf runtime, making it highly versatile for different use cases.

### Loading eBPF Bytecode from ELF Files

llvmbpf supports loading eBPF bytecode directly from ELF files, which is a common format for storing compiled eBPF programs. This feature is particularly useful when working with existing eBPF toolchains.

```c
bpf_object *obj = bpf_object__open(ebpf_elf.c_str());
if (!obj) {
    return 1;
}
std::unique_ptr<bpf_object, decltype(&bpf_object__close)> elf(
    obj, bpf_object__close);

bpf_program *prog;
for ((prog) = bpf_object__next_program((elf.get()), nullptr);
     (prog) != nullptr;
     (prog) = bpf_object__next_program((elf.get()), (prog))) {
    llvmbpf_vm vm;
    vm.load_code((const void *)bpf_program__insns(prog),
                 (uint32_t)bpf_program__insn_cnt(prog) * 8);
}
```

However, the `bpf.o` ELF file has no map and data relocation support. We recommend using bpftime to load and relocate the eBPF bytecode from an ELF file. This includes:

- Writing a loader similar to the kernel eBPF loader to load the eBPF bytecode (see an example [here](https://github.com/eunomia-bpf/bpftime/blob/master/example/xdp-counter/xdp-counter.c)).
- Using libbpf, which supports:
  - Relocation for maps, where the map ID is allocated by the loader and bpftime. You can use the map ID to access maps through the helpers.
  - Accessing data through the `lddw` helper function.
- After loading the eBPF bytecode and completing relocation, you can use the [bpftimetool](https://eunomia.dev/zh/bpftime/documents/bpftimetool/) to dump the map information and eBPF bytecode.

### Maps and Data Relocation Support

llvmbpf offers extensive support for maps and data relocation, allowing developers to write more complex eBPF programs that interact with different data sources. For instance, you can use helper functions to access maps or define maps as global variables in your eBPF programs.

The eBPF can work with maps in two ways:

- Using helper functions to access the maps, like `bpf_map_lookup_elem`, `bpf_map_update_elem`, etc.
- Using maps as global variables in the eBPF program and accessing the maps directly.

```cpp
uint32_t ctl_array[2] = { 0, 0 };
uint64_t cntrs_array[2] = { 0, 0 };

void *bpf_map_lookup_elem(uint64_t map_fd, void *key) {
    if (map_fd == 5) {
        return &ctl_array[*(uint32_t *)key];
    } else if (map_fd == 6) {
        return &cntrs_array[*(uint32_t *)key];
    } else {
        return nullptr;
    }
}
```

### Building into Standalone Binary for Deployment

One of the standout features of llvmbpf is the ability to compile eBPF programs into standalone binaries. This makes it possible to deploy eBPF applications in environments where installing dependencies is not feasible, such as microcontrollers or other embedded systems.

You can build the eBPF program into a standalone binary that does not rely on any external libraries and can be executed like normal C code with helper and map support.

This approach offers several benefits:

- Easily deploy the eBPF program to any machine without needing to install dependencies.
- Avoid the overhead of loading the eBPF bytecode and maps at runtime.
- Make it suitable for microcontrollers or embedded systems that do not have an OS.

Here’s a basic example:

```c
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

int bpf_main(void* ctx, uint64_t size);

uint32_t ctl_array[2] = { 0, 0 };
uint64_t cntrs_array[2] = { 0, 0 };

void *_bpf_helper_ext_0001(uint64_t map_fd, void *key) {
  printf("bpf_map_lookup_elem %lu\n", map_fd);
  if (map_fd == 5) {
    return &ctl_array[*(uint32_t *)key];
  } else if (map

_fd == 6) {
    return &cntrs_array[*(uint32_t *)key];
  } else {
    return NULL;
  }
}

void* __lddw_helper_map_val(uint64_t val) {
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
    return 0;
}
```

Compile the C code with the LLVM IR:

```sh
clang -g main.c xdp-counter.ll -o standalone 
```

You can then run the `standalone` eBPF program directly. Compared to native C code development, this ensures that the eBPF part is verified after integration with the verifier.

## Optimization Techniques

llvmbpf provides several optimization techniques to enhance the performance of eBPF programs. Two notable methods include:

### Inlining Maps and Helper Functions

By inlining maps and helper functions, llvmbpf reduces the overhead of function calls, enabling more efficient execution of eBPF programs.

```sh
clang -S -O3 -emit-llvm libmap.c -o libmap.ll
llvm-link -S -o xdp-counter-inline.ll xdp-counter.ll libmap.ll
opt --always-inline -S xdp-counter-inline.ll -o xdp-counter-inline.ll
```

### Using Original LLVM IR from C Code

Instead of relying solely on eBPF instructions, llvmbpf allows developers to use original LLVM IR generated from C code. This flexibility opens the door for more advanced optimizations and higher performance.

```c
int bpf_main(void* ctx, int size) {
    _bpf_helper_ext_0006("hello world: %d\n", size);
    return 0;
}
```

eBPF is an instruction set designed for verification, but it may not be the best for performance. llvmbpf also supports using the original LLVM IR from C code. See [example/load-llvm-ir](https://github.com/eunomia-bpf/llvmbpf/tree/main/example/load-llvm-ir) for an example. You can:

- Compile the C code to eBPF for verification.
- Compile the C code to LLVM IR and native code for execution in the VM.

## Conclusion

llvmbpf is a powerful tool for developers looking to leverage eBPF outside the kernel. With features like AOT compilation, standalone deployment, and extensive support for maps and relocation, it offers a flexible and high-performance solution for a wide range of use cases. Whether you’re working on networking, security, or performance monitoring applications, llvmbpf provides the tools you need to build efficient and portable eBPF programs.

## Links

- [llvmbpf](https://github.com/eunomia-bpf/llvmbpf)
- [bpftime](https://github.com/eunomia-bpf/bpftime)
- [LLVM BPF Examples](https://github.com/eunomia-bpf/llvmbpf/tree/main/example)
- [eBPF ISA](https://www.ietf.org/archive/id/draft-ietf-bpf-isa-00.html)
