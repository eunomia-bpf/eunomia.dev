# talk draft in kubecon

## IntroductionOpening (1 minute)

Welcome, everyone.

Thank you for joining me to discuss the innovative intersection of eBPF and WebAssembly—technologies revolutionizing observability within our systems.

Combining eBPF with Wasm provides a robust solution for non-intrusive deployment and advanced security checks within Kubernetes pods. Today, we'll explore how these technologies can be leveraged for efficient and secure software development. Let's dive into a detailed discussion on their benefits, challenges, and the future they hold.

My name is Yusheng Zheng, currently maintaining a small community called eunomia-bpf, building open source projects to make eBPF easier to use, and exploring new technologies and runtimes related to eBPF.

<!-- TOC -->

- [talk draft in kubecon](#talk-draft-in-kubecon)
  - [IntroductionOpening (1 minute)](#introductionopening-1-minute)
  - [Slide 15-17: Faster, easier \& safer eBPF deployment and Trade-offs (4 minutes)](#slide-15-17-faster-easier--safer-ebpf-deployment-and-trade-offs-4-minutes)
  - [Slice 16: Use container tools to run Wasm + eBPF (1 min)](#slice-16-use-container-tools-to-run-wasm--ebpf-1-min)
  - [Slide 17: Developer Experience (1 min)](#slide-17-developer-experience-1-min)
  - [Slide 18: Examples (1 mins)](#slide-18-examples-1-mins)
  - [Slide 19: The Operational Framework of wasm-bpf (1 minutes)](#slide-19-the-operational-framework-of-wasm-bpf-1-minutes)
  - [Slide 20: The wasm-bpf Development Process (1 minutes)](#slide-20-the-wasm-bpf-development-process-1-minutes)
  - [Slide 21: Challenges (2 mins)](#slide-21-challenges-2-mins)
  - [Slide 22: Wasm with user space eBPF (2 minutes)](#slide-22-wasm-with-user-space-ebpf-2-minutes)
  - [Slide 23-26: How eBPF Enhances Wasm Developer Experience (3 minutes)](#slide-23-26-how-ebpf-enhances-wasm-developer-experience-3-minutes)
  - [Closing (1 minute)](#closing-1-minute)

<!-- /TOC -->
  - [Slide 23-26: How eBPF Enhances Wasm Developer Experience (3 minutes)](#slide-23-26-how-ebpf-enhances-wasm-developer-experience-3-minutes)
  - [Closing (1 minute)](#closing-1-minute)

<!-- /TOC -->
## Slide 15-17: Faster, easier & safer eBPF deployment and Trade-offs (4 minutes)

However, there are trade-offs. The migration of libraries and toolchains to this new model is not trivial, with considerations around limited eBPF features in Wasm environments. But the familiar development experience, akin to that provided by libbpf-bootstrap, is a testament to our progress.

## Slice 16: Use container tools to run Wasm + eBPF (1 min)

This is a demo of how to use container tools, like the podman to run Wasm and eBPF. As the GIF shows, we can using a podman container to start and run a eBPF program in WebAssembly, which can trace the run queue (scheduler) latency as a histogram for the linux system. This program is ported from the bcc tool, and compiled into WebAssembly with the eunomia-bpf toolchain. It will fetch data from bpf hash maps in the kernel, do some post-processing in userspace, and then print the result to the console. The eBPF program is integrated into the userspace WebAssembly application, which can be packed into a OCI image and started as a container with the WasmEdge runtime.

We can also list the existing podman WebAssembly containers, and see the container we just started, and stop and remove the container.

## Slide 17: Developer Experience (1 min)

libbpf-bootstrap is a widely used framework for developing eBPF programs in C/C++.
The developer experience in WebAssembly is similar to that of libbpf-bootstrap, includes automatically generating skeleton (bpf code framework) and type definitions, just like the bpftool and libbpf-bootstrap does.

The right part is the auto-generated skeleton from our WebAssembly bpftool, the left part is the userspace code of loading and attaching the eBPF program in C, which will be compiled into WebAssembly.

## Slide 18: Examples (1 mins)

Let's take a moment to walk through some hands-on examples where eBPF in WebAssembly can support.

First, we have 'Uprobe' for Observability or Tracing. This is like setting up a watchtower inside your applications, letting you keep an eye on how functions are running without modify them.

Next is 'XDP' for Networking. This can be used to process packets at the lowest level before they reach the network stack, allowing you to filter, redirect, or drop packets as needed.

And then we have 'LSM' for Security. This allows you to set rules on what the system can and cannot do, like blocking a process from accessing a file or network port in the kernel.

## Slide 19: The Operational Framework of wasm-bpf (1 minutes)

Let's take a look at how wasm-bpf works. The project essentially wants to treat the Wasm sandbox as an alternative user-state runtime space on top of the OS, which means we can use Wasm to develop and deploy eBPF programs as ordinary eBPF programs, but with the added benefits of security and ease of moving from one system to another.

In the runtime, a Wasm module can managing multiple eBPF programs, and allow `dynamically load` eBPF programs from the Wasm sandbox into the kernel, select the desired events to attach them, unattach them, control the complete lifecycle of multiple eBPF objects, and support most eBPF program types.

Communication is a two-way street with wasm-bpf. It sets up a path for back-and-forth conversations with the kernel using eBPF Maps, making data transfer smooth and efficient with ring buffers. The eBPF can send of messages from the kernel state to the user state via `ring buffering` and perf event polling, or access hashmaps from the Wasm virtual machine. The bpf maps can also be accessed with share memory between kernel and Wasm runtime. This setup is not only flexible but also ready to grow with new kernel features without the need to modify the virtual machine's system interface.

## Slide 20: The wasm-bpf Development Process (1 minutes)

Now let's move on to how we create eBPF applications with wasm.

To develop an eBPF program, we first need to compile the corresponding source code into bpf objects using the clang/LLVM toolchain, which contains the bpf bytecode and the corresponding data structure definitions, maps and progs definitions in BTF format. Then, we can use BTF info to generate skeleton and bindings for userspace programs development. The approach is similar to component model in WebAssembly, in which we use wit-bindgen and other tools to generate bindings. Then, user can develop the userspace program in C/C++/Rust/Go, compile it into WebAssembly, and packed it with the eBPF bytecode into a OCI image.

In the Wasm-bpf project, with the support of code generation techniques and BTF (BPF type format) information in the toolchain, all communications between Wasm and eBPF do not need to go through serialization and deserialization mechanisms. At the same time, the eBPF development experience is just like the bpftool and libbpf-bootstrap does.

The lightweight nature of compiled eBPF-Wasm modules, which are typically around 100Kb, and they can be dynamically loaded and executed in 100ms. It's optimized for rapid deployment and execution, aligning perfectly with the fast-paced, dynamic requirements of cloud-native environments.

## Slide 21: Challenges (2 mins)

We have overcome some challenges before we can fully use eBPF's capabilities within Wasm for Kubernetes.

Firstly, we've got some of libraries for C/C++, Rust, and Go, each enabling eBPF interactions in their respective languages. These are libbpf for C/C++, libbpf-rs for Rust, and cilium/go for Go. We need to port these libraries to Wasm to enabling developing eBPF programs in these languages with WebAssembly.

Another challenge is the data layout. The data layout of eBPF programs, which is 64 bit, maybe different from that of Wasm, maybe 32 bit or 64 bit. We need to convert the data layout of kernel eBPF programs to the correct data layout of Wasm programs, when we need to communicate between kernel eBPF programs and userspace Wasm programs.

The last challenge is the kernel compatibility.

The eBPF programs may need a specific kernel version to run and also require enable the kernel configuration. We can use Compile-Once, Run Everywhere (CO-RE) technology to enhance the portability of eBPF programs, but we still need to ensure that the kernel version is compatible with the eBPF programs. We can also look at a userspace eBPF runtimeto run eBPF programs in userspace, or a compatibility layer for different kernel features.

## Slide 22: Wasm with user space eBPF (2 minutes)

Today, we're also examining a new development in system observability and interaction: the combination of WebAssembly, or Wasm, with user space eBPF runtimes.

Wasm with kernel eBPF unlocks a lot of potential, allowing us to engage deeply with the Linux kernel. However, it does come with a need for specific kernel versions supports eBPF, and privileges to load the eBPF into the kernel.

Enter bpftime, our new approach to eBPF that operates entirely in user space. It means we can deploy existing eBPF tracing programs without special permissions and without depending on the kernel's version, and even not limited to Linux system.

bpftime allows us to use eBPF tools with Uprobes and Syscall tracepoints to monitor and trace applications in userspace. These tools are lightweight and don't require stopping or tweaking the applications they monitor.

Uprobe are like the user space counterparts to kprobes, which allow us to trace userspace functions in eBPF. However, the kernel Uprobe is slow due to the overhead of context switching between kernel and user space. bpftime solves this problem by running entirely in user space, with the added advantage of being up to 10 times faster than kernel uprobe.

Plus, bpftime plays nicely with existing eBPF toolchains. It supports inter-process eBPF maps in userspace or interacting with kernel eBPF maps, and you can run familiar tools like bcc and bpftrace entirely in user space without any change to their code.

So, we're looking at a more flexible way to gather insights and manage systems, and also exploring potentially integrating with technologies like DPDK for network tasks. It's a step forward in making powerful system tools more accessible and efficient.

## Slide 23-26: How eBPF Enhances Wasm Developer Experience (3 minutes)

To wrap up our technical discussion, we will explore how eBPF elevates the Wasm development experience, particularly through advanced security mechanisms for WASI and sophisticated tracing capabilities that simplify debugging.

## Closing (1 minute)

In closing, the fusion of Wasm and eBPF is more than just a technological innovation; it's a new frontier in Kubernetes pod deployment, data analytics, and system security. We're excited for you to explore these possibilities and contribute to their evolution. Thank you for joining us today, and we look forward to your questions and insights.
