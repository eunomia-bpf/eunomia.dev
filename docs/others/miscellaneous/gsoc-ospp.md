# GSOC - OSPP draft proposal

## GSOC 2024

Unleash eBPF Potential with our tools and runtimes

Eunomia Lab is an innovative open-source organization dedicated to advancing the eBPF ecosystem. Our mission is to create and improve toolchains and runtimes that enhance the functionality and efficiency of eBPF, a revolutionary technology for running sandboxed programs within the Linux kernel without changing kernel source code or loading kernel modules. Our projects include: 

- bpftime: A userspace eBPF runtime. It offers rapid uprobe 10x faster than kernel uprobes, and syscall hook capabilities. It's compatible with kernel eBPF and existing eBPF toolchains, and can be injected into any running process without restart or manual recompilation. It can work with kernel eBPF or the eBPF runtime in other userspace processes. 
- Wasm-bpf: Cooprated with WaseEdge, we build the first user-space development library, toolchain, and runtime for general eBPF programs based on WebAssembly, allows lightweight Wasm sandboxes to deploy and control eBPF applications in k8s clusters. 
- GPTtrace: The first tool generates eBPF programs and traces the Linux kernel through natural language. With our AI agents, it can produce correct eBPF programs on 80%, while a baseline of GPT-4 is 30%. 
- eunomia-bpf: A tool to help developers build, distribute and run eBPF programs easier with JSON and WebAssembly OCI images 

Our commitment extends beyond tool development to education. We offer extensive resources for those looking to master eBPF, from beginners to advanced users. To discover more about our projects and educational materials, visit us at eunomia.dev.

## OSPP 24

eunomia-bpf 是一个开源组织，致力于于推动 eBPF 生态系统的进步，希望通过创建和改善工具链及运行时来提升 eBPF 技术的功能性和效率，这一技术能在不修改内核代码或加载模块的情况下，在 Linux 内核中运行沙盒化程序。

我们的项目包括：

- bpftime：用户空间 eBPF 运行时，提供比内核快 10 倍的 uprobe 速度，支持 syscall 钩子，无需重启或重新编译即可注入运行中进程。
- Wasm-bpf：与 WaseEdge 合作开发的，基于 WebAssembly 的 eBPF 应用开发工具链和运行时，使得在 k8s 集群中部署轻量级 Wasm 沙盒成为可能。
- GPTtrace：首个通过自然语言生成 eBPF 程序的工具，成功率达 80%。
- eunomia-bpf framework：简化 eBPF 程序开发、分发和运行流程的工具，使用 JSON 和 WebAssembly OCI 镜像。
- eunomia-bpf 也致力于提供 eBPF 学习资源和丰富的教程。更多信息，请访问 eunomia.dev。

