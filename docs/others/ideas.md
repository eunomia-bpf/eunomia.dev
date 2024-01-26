# Possible ideas for the future

## bpftime

An userspace eBPF runtime that allows existing eBPF applications to operate in unprivileged userspace using the same libraries and toolchains. It offers Uprobe and Syscall tracepoints for eBPF, with significant performance improvements over kernel uprobe and without requiring manual code instrumentation or process restarts. The runtime facilitates interprocess eBPF maps in userspace shared memory, and is also compatible with kernel eBPF maps, allowing for seamless operation with the kernel's eBPF infrastructure. It includes a high-performance LLVM JIT for various architectures, alongside a lightweight JIT for x86 and an interpreter.

For more details, see:

- <https://eunomia.dev/bpftime>
- [https://github.com/eunomia-bpf/bpftime](https://github.com/eunomia-bpf/bpftime)

## Porting bpftime to macOS

The goal of this project is to port `bpftime` to macOS, expanding its cross-platform capabilities and enabling macOS users to leverage the powerful features of `bpftime`. This porting effort is crucial for fostering a more inclusive and diverse open source community, enabling developers who use macOS as their primary development environment to contribute to and benefit from `bpftime`.

### Objectives

1. **Compatibility and Integration**: Achieve full compatibility of `bpftime` with macOS, ensuring that core features and capabilities are functional on this platform.
2. **Performance Optimization**: Fine-tune the performance of `bpftime` on macOS, focusing on optimizing the LLVM JIT and the lightweight JIT for x86 specifically for macOS architecture.
3. **Seamless Integration with macOS Ecosystem**: Ensure that `bpftime` integrates smoothly with macOS  environments, providing a native and efficient development experience for macOS eBPF users.
4. **Documentation and Tutorials**: Develop documentation and tutorials tailored to macOS users, facilitating easy adoption and use of `bpftime` on this platform.

### Skills Required

- Proficiency in C/C++ and system programming.
- Familiarity with macOS development environment and tools.
- Understanding of eBPF and its applications.

### Expected Outcomes

- A functional port of `bpftime` for macOS, with all features operational.
- documentation and guides for using `bpftime` on macOS.

### Impact

This project will significantly contribute to the open source community by expanding the reach of `bpftime` to a broader audience. By enabling macOS compatibility, the project opens up opportunities for a wider range of developers to experiment with and contribute to `bpftime`, fostering innovation and collaboration in the realm of userspace eBPF runtime development.

## bpftime + fuse: Exploring the Possibility of Synergistic Acceleration Combining User Space File System with Kernel Space

In modern operating systems, `fuse` (Filesystem in Userspace) has become a popular choice, allowing developers to create file systems in user space without modifying kernel code. However, the cost of system calls still exists. This is where `bpftime` can play a role.

**Core Advantages and Possibilities**:

1. **User Space and Kernel Space Synergistic Optimization**: Through `bpftime`, certain file system operations, such as caching, metadata queries, etc., can be pre-processed in user space and interact with eBPF in kernel space only when really needed. This greatly reduces unnecessary system calls, speeds up file operations, and accelerates possible fast paths, similar to the XRP mechanism.
2. **User Space bypass Fuse Mechanism**: We can implement a complete kernel bypass fuse using eBPF to connect corresponding user space programs with VFS and libfuse, achieving a complete kernel bypass user space file system without the need for invasive changes or linking to user space applications.
3. **Dynamic Adjustment of Strategies**: The user space `bpftime` can dynamically collect performance data and adjust strategies accordingly, such as selecting the best caching strategy, all of which can be done at runtime without the need for shutdown or restart.
4. **Stronger Customization Capability**: Developers can customize specific eBPF programs according to specific application scenarios, to adapt to different workloads and optimization targets.

## bpftime + Userspace Network: Bringing Synergistic Intelligence to Network Communication

Networking is fundamental in modern computing environments, but traditional networking approaches may not meet the demands of modern applications with increasingly complex communication patterns and high traffic needs. In this context, the combination of `bpftime` with network libraries offers a new way to handle network requests.

**Core Advantages and Possibilities**:

1. **Intelligent Network Flow Processing**: `bpftime` allows developers to pre-process network requests in user space, such as traffic balancing, QoS policies, etc., and collaborate with eBPF in kernel space only when necessary, effectively reducing context switches and improving network efficiency.
2. **Dynamic Network Policy Adjustment**: Similar to the collaborative work with file systems, `bpftime` can dynamically collect network performance data and adjust network processing strategies, such as dynamically adjusting traffic distribution, quickly switching communication paths, etc.
3. **Higher Network Throughput**: Pre-processing network requests in user space can avoid the overhead of traditional network stacks, thereby achieving higher network throughput, especially in high traffic and low latency scenarios. User space eBPF can be combined with user space network libraries to accelerate service mesh network request forwarding and processing in scenarios like Cilium.

`bpftime` opens up new possibilities for collaborative work between user space and kernel space, both in file systems and network communication. This synergistic optimization offers an efficient, flexible, and customizable framework to meet the demands of modern complex computing environments. In the future, this collaborative work model could become the standard way of interaction between operating systems and applications.

## Userspace AOT Compilation and Execution of eBPF: New Opportunities for Lightweight Containers

**Background**:

In the fields of cloud-native, IoT, and embedded systems, the three core challenges are limited resources, high security requirements, and the need for highly automated deployment and management. Existing container technologies, while providing high levels of abstraction and isolation, still incur runtime overhead. The potential capabilities of user space eBPF, especially its Ahead-of-Time (AOT) compilation into machine code and signing, bring new possibilities to these fields.

**Goals**:

1. Develop a user space eBPF AOT compilation and execution mechanism, aimed at providing a lightweight, event-driven computing model for embedded and resource-constrained environments.
2. Integrate this technology into FaaS lightweight containers to reduce their startup latency and improve execution efficiency.
3. Design and implement a plugin system that allows eBPF programs to be embedded into other applications, providing dynamic, highly optimized functional extensions.

**Challenges**:

1. **Balancing Performance and Security**: How to ensure the performance of eBPF programs while guaranteeing their complete security in user space, especially in the absence of kernel-level isolation.
2. **Generality and Specificity**: Designing a system that meets the specific needs of embedded and IoT devices while being sufficiently generic to adapt to different application scenarios.
3. **Compatibility and Innovation**: Ensuring that new technologies are compatible with the existing container ecosystem and IoT standards while introducing innovations.

**Possibilities**:

1. **Low Overhead Computing**: The lightweight nature of eBPF and its ability to AOT compile into machine code can significantly reduce computing overhead in resource-constrained environments.
2. **Dynamic Extensibility**: Embedding eBPF programs as plugins into other applications provides highly dynamic, on-demand functionality for various applications.
3. **Cross-Platform and Multi

-Scenario Applications**: The portability and security of eBPF programs make them ideal for a variety of platforms and application scenarios, including edge computing, smart homes, automotive systems, etc.

User space eBPF's AOT compilation and execution offer a unique, highly potential method to meet the challenges and needs of the cloud-native, IoT, and embedded systems fields. Its unique combination of performance, security, and flexibility makes it an important direction for future research and innovation in these fields.
