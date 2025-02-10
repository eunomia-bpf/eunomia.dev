---
date: 2023-02-11
---

# Wasm-bpf: A Common eBPF Kernel Programmability for Cloud-Native Webassembly

> Author: Yusheng Zheng, Mao-Lin Chen

Originally developed with a browser-safe sandbox in mind, Wasm has evolved to make WebAssembly a high-performance, cross-platform and multilingual software sandbox environment for cloud-native software components, and Wasm lightweight containers are well suited as the next-generation serverless platform runtime. Another exciting trend is the rise of eBPF, which enables cloud-native developers to build secure networks, service grids, and multiple observable components, and which is also gradually penetrating and penetrating deeper into kernel components, providing more powerful kernel-state programmable interactions.

Wasm-bpf is a new open source project [1] that defines a set of abstractions for eBPF-related system interfaces and provides a corresponding set of development toolchains, libraries, and generic Wasm + eBPF runtime platform instances, giving applications in any Wasm virtual machine or Wasm lightweight container the ability to sink and extend usage scenarios to the kernel state, accessing almost all data in the kernel state and The eBPF runtime platform instance allows applications in any Wasm virtual machine or Wasm lightweight container to sink and expand their usage scenarios to the kernel state, access almost all data in the kernel state and user state, and achieve programmable control over the entire operating system in many aspects such as networking and security, thus greatly expanding the WebAssembly ecosystem in non-browser application scenarios.
<!-- more -->

## eBPF-based System Interface for Wasm

Perhaps you have also read this quote from Solomon Hykes (one of the founders of Docker).

<!-- TOC -->

- [Wasm-bpf: A Common eBPF Kernel Programmability for Cloud-Native Webassembly](#wasm-bpf-a-common-ebpf-kernel-programmability-for-cloud-native-webassembly)
  - [eBPF-based System Interface for Wasm](#ebpf-based-system-interface-for-wasm)
  - [eBPF: Extending the Kernel Securely and Efficiently](#ebpf-extending-the-kernel-securely-and-efficiently)
    - [The Future of eBPF: A JavaScript-Like Programmable Interface for the Kernel](#the-future-of-ebpf-a-javascript-like-programmable-interface-for-the-kernel)
  - [Interaction flow between user space and eBPF programs](#interaction-flow-between-user-space-and-ebpf-programs)
    - [Common user-state eBPF development framework](#common-user-state-ebpf-development-framework)
    - [A new eBPF development framework defined on top of the user-state Wasm-eBPF system interface](#a-new-ebpf-development-framework-defined-on-top-of-the-user-state-wasm-ebpf-system-interface)
  - [References](#references)

<!-- /TOC -->
## eBPF: Extending the Kernel Securely and Efficiently

eBPF is a revolutionary technology, originating from the Linux kernel, that allows sandboxed programs to be run in the kernel of the operating system. It is used to safely and efficiently extend the functionality of the kernel without changing the kernel's source code or loading kernel modules.

Looking historically, the operating system kernel has been an ideal place to implement various capabilities like observability, security, and networking due to its privileged ability to supervise and control the entire system. However, due to the high demands on stability and security, kernel feature iterations are typically very cautious, and it is difficult to accept customized, less common functionality improvements. Therefore, compared to the functionalities in user space, the rate of innovation at the kernel-level operating system layer has always been relatively low.[2]

<div align="center">
<img src=https://github.com/ebpf-io/ebpf.io-website/blob/main/content/static-pages/what-is-ebpf/overview.png width=60% alt="eBPF Overview" />
</div>

eBPF fundamentally changes this paradigm. By allowing sandboxed programs to run within the operating system, application developers can programmatically add additional functionalities to the operating system at runtime. The operating system then ensures safety and execution efficiency, as if it were compiled locally with the help of a Just-In-Time (JIT) compiler and a verification engine. eBPF programs are portable across kernel versions and can be automatically updated, thus avoiding workload interruptions and node restarts.

Today, eBPF is widely used in various scenarios: in modern data centers and cloud-native environments, it provides high-performance network packet processing and load balancing; with very low resource overhead, it achieves observability of a variety of fine-grained metrics, helping application developers track applications and provide insights for troubleshooting performance issues; it ensures the secure execution of applications and containers, and more. The possibilities are endless, and the innovation unleashed by eBPF in the operating system kernel is just beginning.[3]

### The Future of eBPF: A JavaScript-Like Programmable Interface for the Kernel

For browsers, the introduction of JavaScript's programmability sparked a significant revolution, turning browsers into almost independent operating systems. Now, looking at eBPF: to understand the impact of eBPF on the programmability of the Linux kernel, it is helpful to have a high-level understanding of the structure of the Linux kernel and how it interacts with applications and hardware.[4]

<div align="center">
<img src=https://github.com/ebpf-io/ebpf.io-website/blob/main/content/static-pages/what-is-ebpf/kernel-arch.png width=60% />
</div>

The main purpose of the Linux kernel is to abstract the hardware or virtual hardware and provide a consistent API (system calls) to allow applications to run and share resources. To achieve this, a series of subsystems and layers are maintained to distribute these responsibilities. Each subsystem typically allows some degree of configuration to take into account the different needs of the user. If the desired behavior cannot be configured, changing the kernel is necessary. Historically, changing the kernel's behavior or enabling user-written programs to run in the kernel has presented two options.

| Support a kernel module locally                                                                                                                                         | Write a kernel module                                                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Change the kernel source code and convince the Linux kernel community that such a change is necessary. Wait a few years for a new kernel version to become a commodity. | Fix it regularly, as every kernel version can break it. Risk breaking your Linux kernel due to the lack of security boundaries |

In practice, neither option is commonly used; the former is too costly, and the latter has almost no portability.

With eBPF, there is a new option to reprogram the behavior of the Linux kernel without changing the kernel's source code or loading kernel modules, while guaranteeing a certain degree of consistency and compatibility of behavior, as well as security, between different kernel versions. To achieve this, eBPF programs also need to have a corresponding set of APIs that allow user-defined applications to run and share resources -- in other words, in a sense, the eBPF virtual machine also provides a set of system call-like mechanisms that are available to Wasm virtual machines and user-state applications through the eBPF and user-state communication mechanisms. With the eBPF and user state communication mechanisms, Wasm VMs and user state applications can also gain full access to this set of "system calls", which can programmatically extend the capabilities of traditional system calls on the one hand, and achieve more efficient programmable IO processing on the other.

![new-os](new-os-model.jpg)

As the diagram above shows, today's Linux kernel is evolving into a new kernel model: user-defined applications can execute in both the kernel and user states, with the user state accessing system resources through traditional system calls and the kernel state interacting with various parts of the system through BPF Helper Calls. As of early 2023, there are more than 220 Helper System Interfaces in the eBPF virtual machine in the kernel, covering a very wide range of application scenarios.

It is important to note that BPF Helper Calls and System Calls are not in competition with each other; they have completely different programming models and scenarios where they have performance benefits, and they do not completely replace each other. The situation is similar for the Wasm and Wasi related ecosystems, where a specially designed wasi interface requires a lengthy standardization process but may yield better performance and portability guarantees for user-state applications in specific scenarios, while eBPF provides a fast and flexible solution for extending the system interface while maintaining the sandbox nature and portability.

The eBPF is still in its early stages, but with the ability to interact between the kernel and user state provided by the current eBPF, applications in the Wasm VM can already obtain data and return values (kprobe, uprobe, ...) from almost any function call in the kernel and user state via the Wasm-bpf system interface transformation. ; collect and understand all system calls and obtain packet and socket level data for all network operations at a very low cost (tracepoint, socket...) Add additional protocol analyzers to the network packet processing solution and easily program any forwarding logic (XDP, TC...) ) to meet changing needs without leaving the packet processing environment of the Linux kernel.

Moreover, eBPF has the ability to write data to any address of any process in user space (bpf_probe_write_user[5]), to modify the return value of kernel functions to a limited extent (bpf_override_return[6]), and even to execute some system calls directly in the kernel state [7]; fortunately, eBPF performs a bytecode analysis before loading into the Fortunately, eBPF performs strict security checks on the bytecode before loading it into the kernel to ensure that there are no memory out-of-bounds or other operations, while many features that may expand the attack surface and pose security risks need to be explicitly chosen to be enabled at compile time before the kernel can be used; certain eBPF features can also be explicitly chosen to be enabled or disabled before the Wasm VM loads the bytecode into the kernel to ensure the security of the sandbox.

All of these scenarios do not require leaving the Wasm lightweight container: unlike traditional applications that use Wasm as a data processing or control plug-in, where these steps are implemented by logic outside the Wasm VM, it is now possible to achieve complete control and interaction with eBPF and almost all system resources that eBPF can access, even generating eBPF in real time, from within the Wasm lightweight container code to change the behavior logic of the kernel, enabling programmability of the entire system from the user state to the kernel state.

## Interaction flow between user space and eBPF programs

eBPF programs are function-based and event-driven, and a specific eBPF program is run when a kernel or user space application passes a hook point. To use an eBPF program, we first need to compile the corresponding source code into bpf bytecode using the clang/LLVM toolchain, which contains the corresponding data structure definitions, maps and progs definitions. progs are program segments, and maps can be used to store data or for bidirectional communication with the user space. After that, we can implement a complete eBPF application with the help of the user state development framework and the loading framework.

### Common user-state eBPF development framework

For a complete eBPF application, there are usually two parts: the user state and the kernel state.

- The user state program needs to interact with the kernel through a series of system calls (mainly bpf system calls), create a corresponding map to store data in the kernel state or to communicate with the user state, dynamically select different segments to load according to the configuration, dynamically modify the bytecode or configure the parameters of the eBPF program, load the corresponding bytecode information into the kernel, ensure security through validators, and communicate with the kernel through maps and the kernel, passing data from the kernel state to the user state (or vice versa) through mechanisms such as ring buffer / perf buffer.
- The kernel state is mainly responsible for the specific computational logic and data collection.

<div align="center">
<img src=https://github.com/ebpf-io/ebpf.io-website/blob/main/content/static-pages/what-is-ebpf/libbpf.png width=60% />
</div>

### A new eBPF development framework defined on top of the user-state Wasm-eBPF system interface

The project essentially wants to treat the Wasm sandbox as an alternative user-state runtime space on top of the OS, allowing Wasm applications to implement the same programming model and execution logic in the sandbox as eBPF applications that normally run in the user state.

Wasm-bpf would require a runtime module built on top of the host (outside the sandbox), and some runtime libraries compiled to Wasm bytecode inside the sandbox to provide complete support.

![wasm](wasm-bpf-no-bcc.png)

To achieve a complete development model, we need.

- a Wasm module can correspond to multiple eBPF procedures.
- an instance of an eBPF procedure can also be shared by multiple Wasm modules
- The ability to dynamically load eBPF programs from the Wasm sandbox into the kernel, select the desired mount points to mount them, unmount them, control the complete lifecycle of multiple eBPF bytecode objects, and support most eBPF program types.
- Bi-directional communication with the kernel via multiple types of Maps, with support for most types of Maps.
- Efficient sending of messages from the kernel state to the user state (and vice versa for ring buffering) via ring buffering and perf event polling.
- It can be adapted to almost any application scenario that uses eBPF programs, and can evolve and extend as kernel features are added, without requiring changes to the Wasm VM's system interface.

This is what the Wasm-bpf project is currently working on. We have also proposed a new Proposal for WASI: WASI-eBPF [7].

In the Wasm-bpf project, all communications between Wasm and eBPF VMs do not need to go through serialization and deserialization mechanisms, and with the support of code generation techniques and BTF (BPF type format [12]) information in the toolchain, we can achieve correct communication between eBPF and Wasm with potentially different structure in vivo layouts, different size end mechanisms, different pointer widths The data can be copied directly from the kernel state to the memory of the Wasm VM when communicating through eBPF Maps, avoiding the extra loss caused by multiple copies. At the same time, the eBPF-Wasm development experience for user-state programs is greatly improved by automatically generating skeleton (bpf code framework) and type definitions.

Thanks to the CO-RE (Compile-Once, Run Everywhere) technology provided by libbpf, porting eBPF bytecode objects between different kernel versions does not introduce an additional recompilation process, nor is there any LLVM/Clang dependency at runtime [12].

Typically a compiled eBPF-Wasm module is only about 90Kb and can be dynamically loaded into the kernel and executed in less than 100ms. We also provide several examples in our repository, corresponding to various scenarios such as observable, network, and security.

We would like to thank Associate Professor Xiaozheng Lai from South China University of Technology, Professor Lijun Chen's team from Xi'an University of Posts and Telecommunications, and teachers Pu Wang and Jicheng Shi from Datan Technology for their guidance and help in combining Wasm and eBPF. blog, we will give a more detailed analysis of the principle and performance, as well as some code examples.

The Wasm-bpf compilation toolchain and runtime modules are currently developed and maintained by the eunomia-bpf open source community, and we thank the PLCT Lab of the Institute of Software of the Chinese Academy of Sciences for their support and funding, and our fellow community members for their contributions. Next, we will also improve and explore more on the corresponding eBPF and Wasm related toolchain and runtime, and actively feed back and contribute to the upstream community.

## References

- [1] wasm-bpf Github open source address: <https://github.com/eunomia-bpf/wasm-bpf>
- [2] When Wasm meets eBPF: Writing, distributing, loading and running eBPF programs using WebAssembly: <https://zhuanlan.zhihu.com/p/573941739>
- [3] <https://ebpf.io/>
- [4] What is eBPF: <https://ebpf.io/what-is-ebpf>
- [5] Offensive BPF: Understanding and using bpf_probe_write_user <https://embracethered.com/blog/posts/2021/offensive-bpf-libbpf-bpf>
- [6] Cloud Native Security Attack and Defense｜Analysis and practice of escape container technology using eBPF: <https://security.tencent.com/index.php/blog/msg/206>
- [7] kernel-versions.md: <https://github.com/iovisor/bcc/blob/master/docs/kernel-versions.md>
- [8] WebAssembly: Docker without containers: <https://zhuanlan.zhihu.com/p/595257541>
- [9] Introduction to WebAssembly, a tool for scalability in cloud-native projects <https://mp.weixin.qq.com/s/fap0bl6GFGi8zN5BFLpkCw>
- [10] WASI-eBPF: <https://github.com/WebAssembly/WASI/issues/513>
- [11] BPF BTF Explained: <https://www.ebpf.top/post/kernel_btf/>
- [12] BPF portability and CO-RE (compile once, run everywhere): <https://cloud.tencent.com/developer/article/1802154>
