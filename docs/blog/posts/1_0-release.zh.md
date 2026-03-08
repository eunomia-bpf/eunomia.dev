---
date: 2024-02-11
---

# eunomia-bpf v1.0：eBPF + Wasm 质的飞跃

随着技术的不断发展，eBPF已经成为了现代Linux内核中的一个核心组件，为开发者提供了强大的性能监控和网络跟踪功能。eunomia-bpf 作为一个新的 eBPF 开源开发框架，旨在简化 eBPF 程序的构建和分发，同时引入了 Wasm技术，为开发者提供了更多的可能性。在过去的半年中，从最初的PoC版本到如今的1.0版本，它已经经历了巨大的变革，成为了一个功能丰富的成熟产品。
<!-- more -->

## eunomia-bpf简介

eunomia-bpf的目标是简化和增强eBPF的使用体验。它结合了CO-RE技术和WebAssembly技术，为开发者提供了一系列强大的工具和功能：

- **简化eBPF程序编写**：自动从内核采样数据并在用户空间打印；自动生成并配置eBPF程序的命令行参数；支持BCC和libbpf风格的内核部分编写。
  
- **使用Wasm构建eBPF程序**：支持C/C++、Rust、Go等多种语言，涵盖从追踪、网络到安全的各种用例。

- **简化eBPF程序分发**：提供工具从云或URL运行eBPF程序，无需重新编译，独立于内核版本和架构；动态加载eBPF程序，支持JSON配置文件或Wasm模块。

## v1.0版本的亮点

### 对更多架构的支持

在1.0版本中，eunomia-bpf框架下的全部工具（包括ecc和ecli）均增加了对`aarch64`架构的支持。在`aarch64`下，`eunomia-bpf`提供与`x86_64`下完全一致的功能。

- 由于我们只有x86_64架构的构建服务器，所以我们使用交叉编译来构建aarch64架构的可执行文件和docker镜像。相关的构建脚本与workflow文件均可以在我们的仓库中找到。

### 更少的外部依赖

在1.0版本中，我们使用`AppImage`来打包发布`ecli`和`ecc`的二进制文件，这也意味着：

- 所有的依赖库都被打包在`ecli`与`ecc`的二进制文件内，也就意味着用户不再需要手动解决`glibc`、`libclang`等库的版本问题，有效避免了不同发行版或不同版本之间的依赖库版本差异问题。
- 发布的`ecli`和`ecc`为静态链接了`libfuse`的AppImage二进制文件，也就意味着发布的`ecli`、`ecc`可以在任何发行版上运行，不依赖于本地提供的`glibc`（只要内核支持fuse）

构建带有所有依赖库的AppImage的workflow文件在仓库中可以找到。

### 对更多附加类型的支持

在`1.0`中，我们增加了对以下eBPF程序类型的支持：

- tc，用以对流量控制进行监控
- xdp，用以对XDP相关数据包进行监控
- profile，用以对某个处理器核心上的内核栈和用户态栈进行监控。

对于这些附加类型，我们均提供相对应的测试和例子。

### 更完善的对C/S架构的支持

我们新增了`ecli-server`，这个项目的作用是通过OpenAPI向远程提供在本机运行ecli程序的能力。运行`ecli-server`后，用户可以通过自己编写的程序使用其所提供的OpenAPI接口运行程序并获取日志，或者通过`ecli`像在本地运行程序一样，在远程运行程序。

### 拆分的bpf-compatible

在之前的发布版中，eunomia-bpf已经支持不依赖于本地BTF的情况下的跨内核版本执行，即tar打包格式。在1.0中，这一功能得到增强：

- 此部分功能被拆分为一个单独的项目，<https://github.com/eunomia-bpf/bpf-compatible>
- 支持基于btfhub将不同发行版、不同内核版本的BTF文件进行裁剪，并将裁剪后的压缩包嵌入到发布的可执行文件内，而后通过特定的API来使用携带的BTF

## 如何开始使用eunomia-bpf

对于那些对eunomia-bpf感兴趣的开发者，我们提供了一系列的资源和工具来帮助您快速上手。

### 资源与示例

- **GitHub模板**：我们为您提供了一个模板，您可以在此基础上开始您的eBPF项目。请访问[eunomia-bpf/ebpm-template](https://github.com/eunomia-bpf/ebpm-template)。
  
- **eBPF程序示例**：为了帮助您更好地理解如何使用eunomia-bpf，我们提供了一些示例程序。您可以在此查看：[examples/bpftools](https://github.com/eunomia-bpf/eunomia-bpf/tree/main/examples/bpftools/)。

- **教程**：如果您是eBPF的新手，我们还为您提供了一个详细的教程，帮助您从零开始。请查看[eunomia-bpf/bpf-developer-tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial)。

### 命令行工具与服务器模式

eunomia-bpf提供了一个命令行工具，使您能够轻松地从云端运行预编译的eBPF程序。只需一行bash命令，您就可以开始：

```bash
# 从以下链接下载最新版本：https://github.com/eunomia-bpf/eunomia-bpf/releases/latest/download/ecli
$ wget https://aka.pw/bpf-ecli -O ecli && chmod +x ./ecli
$ sudo ./ecli run https://eunomia-bpf.github.io/eunomia-bpf/sigsnoop/package.json
```

此外，您还可以使用服务器模式来管理和动态安装eBPF程序。启动服务器后，您可以使用ecli来控制远程服务器并管理多个eBPF程序：

```console
$ sudo ./ecli-server
[2023-08-08 02:02:03.864009 +08:00] INFO [server/src/main.rs:95] Serving at 127.0.0.1:8527
```

可以使用 ecli 控制 server 并管理多个 eBPF 程序：

```console
$ ./ecli client start sigsnoop.json # start the program
1
$ ./ecli client log 1 # get the log of the program
TIME     PID    TPID   SIG    RET    COMM   
02:05:58  79725 78132  17     0      bash
02:05:59  77325 77297  0      0      node
02:05:59  77297 8042   0      0      node
02:05:59  77297 8042   0      0      node
02:05:59  79727 79726  17     0      which
02:05:59  79726 8084   17     0      sh
02:05:59  79731 79730  17     0      which
```

更多关于服务器模式的信息，请查看[documents/src/ecli/server.md](https://github.com/eunomia-bpf/eunomia-bpf/blob/master/documents/src/ecli/server.md)。


## 总结

eunomia-bpf v1.0在过去的半年中经历了从初步的PoC到现在的成熟版本的转变。这个版本不仅增加了对多种架构的支持，还简化了外部依赖，使得eBPF程序的构建和分发变得更加简单。同时，通过引入Wasm技术，eunomia-bpf为开发者打开了新的大门，使得eBPF程序的开发变得更加灵活。我们期待这个框架能为Linux社区带来更多的便利，并期待其未来的发展。更多的信息请参考我们的 Github 仓库：<https://github.com/eunomia-bpf/eunomia-bpf>
