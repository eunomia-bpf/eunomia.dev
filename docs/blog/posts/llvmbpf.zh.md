---
date: 2024-09-10
---
# 构建高性能的用户态 eBPF 虚拟机：基于 LLVM 的 llvmbpf 项目

我们很高兴向大家介绍 [**llvmbpf**](https://github.com/eunomia-bpf/llvmbpf)，这是一个全新的项目，旨在为开发者提供一个高性能、支持多架构的 eBPF 虚拟机（VM）。llvmbpf 利用 LLVM 框架实现了即时编译（JIT）和提前编译（AOT），让你能够在用户态中高效运行 eBPF 程序。

该项目是 [bpftime](https://github.com/eunomia-bpf/bpftime) 项目的一部分，但它专注于核心虚拟机功能。llvmbpf 可以作为独立的 eBPF VM 库或编译工具使用。它经过性能优化，具备极高的灵活性，并且依赖极少，能够轻松集成到各种环境中而不会带来额外负担。
<!-- more -->

## 为什么选择 llvmbpf？

虽然已有许多用户态 eBPF 运行时，但我们开发 llvmbpf 是为了解决一些现有解决方案可能无法满足的特定需求：

1. **AOT 编译器**：llvmbpf 能够将 eBPF 字节码编译成本机 ELF 目标文件，方便开发者部署预编译的 eBPF 程序。这种方法在资源受限的环境中特别高效。此外，你还可以基于 LLVM IR 实验不同的优化技术，获得更多的灵活性和对编译过程的控制。

2. **独立部署**：通过 llvmbpf，你可以将 eBPF 程序构建为无需外部依赖的独立二进制文件。这一功能在嵌入式系统、微控制器或其他难以安装额外软件的环境中非常实用。相比于原生 C 代码开发，llvmbpf 确保了 eBPF 部分在集成到验证器后得到了验证。

3. **多架构支持**：llvmbpf 设计为兼容多种架构，能够在不同的硬件平台上通用。

4. **Maps 和数据重定位支持**：与许多其他用户态 eBPF 解决方案不同，llvmbpf 提供了对 eBPF maps、数据重定位以及 `lddw` 辅助函数的强大支持，使得你可以开发更复杂、更强大的 eBPF 程序。

5. **可扩展的优化方法**：利用 LLVM 的强大优化能力，llvmbpf 支持高级优化，例如内联 maps 和辅助函数，以及使用原始的 LLVM IR 来提升性能。

接下来，我们将通过一些实际示例来展示如何使用 llvmbpf，重点介绍其核心功能和特点。

如果你需要一个全面的用户态 eBPF 运行时，支持 maps、辅助函数，并能够无缝执行 Uprobe、系统调用跟踪、XDP 等 eBPF 程序——与内核功能类似但运行在用户态——可以参考 [bpftime](https://github.com/eunomia-bpf/bpftime) 项目。

## 开始使用 llvmbpf

### 将 llvmbpf 作为库使用

llvmbpf 可以作为库在你的应用程序中使用，以加载和执行 eBPF 程序。以下是一个基本示例：

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

这个示例展示了如何加载 eBPF 字节码、注册外部函数并在虚拟机中执行程序。

### 将 llvmbpf 作为 AOT 编译器使用

llvmbpf 最强大的功能之一是它可以作为 AOT 编译器，将 eBPF 字节码转换为本机 ELF 目标文件。这种方法不仅提升了性能，还简化了 eBPF 程序的部署。

你可以使用命令行工具从 eBPF 字节码生成 LLVM IR：

```console
# ./build/cli/bpftime-vm build .github/assets/sum.bpf.o -emit-llvm > test.bpf.ll
# opt -O3 -S test.bpf.ll -opaque-pointers -o test.opt.ll
# cat test.opt.ll
```

将 eBPF 程序 AOT 编译为 ELF 文件：

```console
# ./build/cli/bpftime-vm build .github/assets/sum.bpf.o
[info] Processing program test
[info] Program test written to ./test.o
```

加载并运行 AOT 编译的 eBPF 程序：

```console
# echo "AwAAAAEAAAACAAAAAwAAAA==" | base64 -d > test.bin
# ./build/cli/bpftime-vm run test.o test.bin
[info] LLVM-JIT: Loading aot object
[info] Program executed successfully. Return value: 6
```

生成的 ELF 目标文件可以与其他目标文件链接，或直接加载到 llvmbpf 运行时，使其在不同场景中都非常实用。

### 从 ELF 文件加载 eBPF 字节码

llvmbpf 支持直接从 ELF 文件加载 eBPF 字节码，这是一种存储已编译 eBPF 程序的常见格式。当你使用现有的 eBPF 工具链时，这一功能特别有用。

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

然而，`bpf.o` ELF 文件不支持 maps 和数据重定位。我们建议使用 bpftime 从 ELF 文件加载并重定位 eBPF 字节码。这包括：

- 编写类似内核 eBPF 加载器的加载器来加载 eBPF 字节码（可以参考[这里](https://github.com/eunomia-bpf/bpftime/blob/master/example/xdp-counter/xdp-counter.c)的一个例子）。
- 使用 libbpf，它支持：
  - maps 的重定位，映射 ID 由加载器和 bpftime 分配。你可以通过辅助函数使用映射 ID 来访问 maps。
  - 通过 `lddw` 辅助函数访问数据。
- 在加载 eBPF 字节码并完成重定位后，你可以使用 [bpftimetool](https://eunomia.dev/zh/bpftime/documents/bpftimetool/) 导出 maps 信息和 eBPF 字节码。

### 支持 eBPF Maps 和数据重定位

llvmbpf 提供了对 maps 和数据重定位的广泛支持，使开发者能够编写与不同数据源交互的复杂 eBPF 程序。例如，你可以使用辅助函数访问 maps，或者在 eBPF 程序中将 maps 定义为全局变量并直接访问。

eBPF 程序可以通过两种方式与 maps 交互：

- 使用辅助函数访问 maps，例如 `bpf_map_lookup_elem`、`bpf_map_update_elem` 等。
- 在 eBPF 程序中将 maps 定义为全局变量，并直接访问这些 maps。

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

### 构建独立二进制文件进行部署

llvmbpf 的一个重要特性是可以将 eBPF 程序编译为独立的二进制文件。这使得在无法安装依赖项的环境中部署 eBPF 应用程序成为可能，例如微控制器或其他嵌入式系统。

你可以将 eBPF 程序构建为独立的二进制文件，无需任何外部库，并

且可以像普通 C 代码一样执行，且支持辅助函数和 maps。

这种方法有以下几个好处：

- 可以轻松地将 eBPF 程序部署到任何机器上，而无需安装依赖项。
- 避免在运行时加载 eBPF 字节码和 maps 的开销。
- 适用于没有操作系统的微控制器或嵌入式系统。

以下是一个基本示例：

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
  } else if (map_fd == 6) {
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

使用 LLVM IR 编译 C 代码：

```sh
clang -g main.c xdp-counter.ll -o standalone 
```

然后你可以直接运行 `standalone` eBPF 程序。与原生 C 代码开发相比，这确保了 eBPF 部分在与验证器集成后得到了验证。

## 优化技术

llvmbpf 提供了多种优化技术来提升 eBPF 程序的性能。两个值得注意的方法包括：

### 内联 Maps 和辅助函数

通过内联 maps 和辅助函数，llvmbpf 减少了函数调用的开销，从而实现更高效的 eBPF 程序执行。

```sh
clang -S -O3 -emit-llvm libmap.c -o libmap.ll
llvm-link -S -o xdp-counter-inline.ll xdp-counter.ll libmap.ll
opt --always-inline -S xdp-counter-inline.ll -o xdp-counter-inline.ll
```

### 使用原始的 LLVM IR 从 C 代码生成

与仅依赖 eBPF 指令集不同，llvmbpf 允许开发者使用从 C 代码生成的原始 LLVM IR。这种灵活性为更高级的优化和更高的性能打开了大门。

```c
int bpf_main(void* ctx, int size) {
    _bpf_helper_ext_0006("hello world: %d\n", size);
    return 0;
}
```

eBPF 是为验证而设计的指令集，但它可能并不是性能最佳的选择。llvmbpf 还支持使用来自 C 代码的原始 LLVM IR。请参见 [example/load-llvm-ir](https://github.com/eunomia-bpf/llvmbpf/tree/main/example/load-llvm-ir) 获取示例。你可以：

- 将 C 代码编译为 eBPF 以进行验证。
- 将 C 代码编译为 LLVM IR 和本机代码，以在 VM 中执行。

## 结论

llvmbpf 是一个强大的工具，适合那些希望在内核之外利用 eBPF 的开发者。通过 AOT 编译、独立部署以及对 maps 和重定位的广泛支持，它为各种应用场景提供了灵活且高性能的解决方案。无论你从事的是网络、安全还是性能监控领域，llvmbpf 都能为你提供构建高效、可移植 eBPF 程序所需的工具。

## 链接

- [llvmbpf](https://github.com/eunomia-bpf/llvmbpf)
- [bpftime](https://github.com/eunomia-bpf/bpftime)
- [LLVM BPF 示例](https://github.com/eunomia-bpf/llvmbpf/tree/main/example)
- [eBPF ISA](https://www.ietf.org/archive/id/draft-ietf-bpf-isa-00.html)
