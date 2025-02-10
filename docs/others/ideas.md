# Possible ideas for the future

This is some possible ideas for open source events, like GSOC(Google Summer of Code) or OSPP(Open Source Promotion Plan) and others. Our projects are designed to suit contributors with varying levels of expertise, from students to more advanced developers.

It's also part of our project roadmap, if you don't participate in these events, you can also help or colaborate with these ideas! Need help? Please contact in the [email list](mailto:team@eunomia.dev) or in the [Discord channel](https://discord.gg/jvM73AFdB8).

## Table of contents

- [Possible ideas for the future](#possible-ideas-for-the-future)
  - [Table of contents](#table-of-contents)
  - [Porting bpftime to macOS and other platforms](#porting-bpftime-to-macos-and-other-platforms)
    - [Objectives for enable eBPF on macOS](#objectives-for-enable-ebpf-on-macos)
    - [Expected Outcomes](#expected-outcomes)
    - [Prerequisites and Skills](#prerequisites-and-skills)
    - [Reference and issue](#reference-and-issue)
  - [VirtIO devices memory address translation fastpath](#virtio-devices-memory-address-translation-fastpath)
    - [Project Overview](#project-overview)
    - [Objectives](#objectives)
    - [Expected Outcomes](#expected-outcomes-1)
    - [Prerequisites and Skills](#prerequisites-and-skills-1)
    - [Reference and Issue](#reference-and-issue-1)
  - [User-Space eBPF Security Modules for Comprehensive Security Policies](#user-space-ebpf-security-modules-for-comprehensive-security-policies)
    - [Project Overview](#project-overview-1)
    - [Objectives](#objectives-1)
    - [Expected Outcomes](#expected-outcomes-2)
    - [Prerequisites and Skills](#prerequisites-and-skills-2)
    - [Reference and Issue](#reference-and-issue-2)
  - [Add Fuzzer and kernel eBPF test for bpftime to improve compatibility](#add-fuzzer-and-kernel-ebpf-test-for-bpftime-to-improve-compatibility)
    - [Project Overview](#project-overview-2)
    - [Timeframe and Difficulty](#timeframe-and-difficulty)
    - [Mentors](#mentors)
    - [Objectives](#objectives-2)
    - [Expected Outcomes](#expected-outcomes-3)
    - [Prerequisites and Skills](#prerequisites-and-skills-3)
    - [Reference and Issue](#reference-and-issue-3)
  - [Living patching distributed RocksDB with shared IO and Network Interface over io\_uring](#living-patching-distributed-rocksdb-with-shared-io-and-network-interface-over-io_uring)
    - [Project Overview](#project-overview-3)
    - [Objectives](#objectives-3)
    - [Expected Outcomes](#expected-outcomes-4)
    - [Prerequisites and Skills](#prerequisites-and-skills-4)
    - [Reference and Issue](#reference-and-issue-4)
  - [Userspace AOT Compilation of eBPF for Lightweight Containers](#userspace-aot-compilation-of-ebpf-for-lightweight-containers)
    - [Overview](#overview)
    - [Goals and Objectives](#goals-and-objectives)
    - [Prerequisites and Skills Required](#prerequisites-and-skills-required)
    - [Expected Outcomes](#expected-outcomes-5)
    - [Additional Resources](#additional-resources)
  - [Userspace eBPF for Userspace File System](#userspace-ebpf-for-userspace-file-system)
    - [Objectives](#objectives-4)
    - [Expected Outcomes](#expected-outcomes-6)
    - [Prerequisites and Skills](#prerequisites-and-skills-5)
    - [Resources](#resources)
  - [BPFTime Profiling and Machine Learning Prediction for far memory or distributed shared memory management](#bpftime-profiling-and-machine-learning-prediction-for-far-memory-or-distributed-shared-memory-management)
    - [Project Overview](#project-overview-4)
    - [Objectives](#objectives-5)
    - [Expected Outcomes](#expected-outcomes-7)
    - [Prerequisites and Skills](#prerequisites-and-skills-6)
    - [Reference and Issue](#reference-and-issue-5)
  - [Large Language Model specific metrics observability in BPFTime](#large-language-model-specific-metrics-observability-in-bpftime)
    - [Project Overview](#project-overview-5)
    - [Objectives](#objectives-6)
    - [Expected Outcomes](#expected-outcomes-8)
    - [Prerequisites and Skills](#prerequisites-and-skills-7)
    - [Reference and Issue](#reference-and-issue-6)
  - [Porting bpftime to Windows, FreeBSD, or other platforms](#porting-bpftime-to-windows-freebsd-or-other-platforms)

For more details, see:

- <https://eunomia.dev/bpftime>
- [https://github.com/eunomia-bpf/bpftime](https://github.com/eunomia-bpf/bpftime)

## Porting bpftime to macOS and other platforms

Since bpftime can run in userspace and does not require kernel eBPF, why not enable eBPF on MacOS/FreeBSD/Other Platforms?

The goal of this project is to port `bpftime` to macOS and other platforms, expanding its cross-platform capabilities and enabling macOS users to leverage the powerful features of `eBPF` in their development and production environments. With bpftime, now you may be able to run bcc and bpftrace tools on macOS and other OSs!

- time: ~175 hour
- Difficulty Level: medium
- mentor: Tong Yu (<yt.xyxx@gmail.com>) and Yuxi Huang (<Yuxi4096@gmail.com>)

### Objectives for enable eBPF on macOS

1. **Compatibility and Integration**: Achieve compatibility of `bpftime` with macOS and/or other OSs, ensuring that core features and capabilities are functional on this platform.
2. **Performance Optimization**: Fine-tune the performance of `bpftime` on macOS and/or other OSs, focusing on optimizing the LLVM JIT and the lightweight JIT for x86 specifically for macOS architecture.
3. **Seamless Integration with macOS Ecosystem**: Ensure that `bpftime` integrates smoothly with macOS and/or other OSs environments, providing a native and efficient development experience for eBPF users.
4. **Documentation and Tutorials**: Develop documentation and tutorials tailored to macOS users, facilitating easy adoption and use of `bpftime` on this platform.

### Expected Outcomes

- A functional port of `bpftime` for macOS and/or other OSs, with core features operational.
- You should be able to run `bpftrace` and `bcc` tools on them, and get expected output.
- documentation and guides for using `bpftime` on macOS and/or other OSs.

### Prerequisites and Skills

- Proficiency in C/C++ and system programming.
- Familiarity with macOS development environment and tools.
- Understanding of eBPF and its applications.

### Reference and issue

- Issue and some initial discussion: <https://github.com/eunomia-bpf/bpftime/issues>
- Some previous efforts: [Enable bpftime on arm](https://github.com/eunomia-bpf/bpftime/pull/151)


## VirtIO devices memory address translation fastpath

The triple address translation from physical VirtIO to the userspace memory is a performance bottleneck. It requires the DPA to HPA to physical memory translation. The VirtIO devices memory address translation fastpath project aims to develop a fastpath for VirtIO devices memory address translation, reducing the overhead of the triple address translation and improving the performance of VirtIO devices. Also, the side channel attack increases the threats for core isolation for the Cloud Vendors. Leveraging BPFTime to design a safe fastpath primitive message passing for dedicated application access to VirtIO devices memory address translation enables safe, efficient and low-latency memory access for VirtIO devices.

### Project Overview

- Time Cost: ~350 hours
- Difficulty Level: Hard
- Mentors: Yiwei Yang (<yyang363@ucsc.edu>) Yusheng Zheng (<mailto:yunwei356@gmail.com>)

### Objectives

- Develop a fastpath for VirtIO devices memory address translation, reducing the overhead of the triple address translation and improving the performance of VirtIO devices.
- Design a safe fastpath primitive message passing for dedicated application access to VirtIO devices memory address translation.

### Expected Outcomes

- A fastpath for VirtIO devices memory address translation, reducing the overhead of the triple address translation and improving the performance of VirtIO devices.
- A safe fastpath primitive message passing for dedicated application access to VirtIO devices memory address translation.

### Prerequisites and Skills

- Proficiency in C/C++ and system programming.
- Understanding of VirtIO devices, micro kernel and memory address translation.
- Familiarity with user-space and kernel-space programming paradigms.
- Experience with developing and testing eBPF programs is highly advantageous.

### Reference and Issue

- Recent paper about offloading userspace program to kernel NVMe device [XRP](https://www.usenix.org/system/files/osdi22-zhong_1.pdf)
- Paper about the fastpath and offloading for DPU [DPFS](https://github.com/IBM/DPFS), [M3](https://os.inf.tu-dresden.de/papers_ps/asmussen-m3-asplos16.pdf)
- Youtube video about the fastpath for VirtIO [VirtIO](https://www.youtube.com/watch?v=nTMls33dG8Q)

## User-Space eBPF Security Modules for Comprehensive Security Policies

### Project Overview

bpftime is a user-space eBPF runtime that allows existing eBPF applications to run directly in unprivileged user space, using the same libraries and toolchains, and to obtain trace analysis results. It provides tracing points such as Uprobe and Syscall tracepoint for eBPF, reducing the overhead by about 10 times compared to kernel uprobe, without the need for manual code instrumentation or process restarts. It enables non-intrusive analysis of source code and compilation processes. It can also be combined with DPDK to implement XDP functionality in user-space networking, compatible with kernel XDP. The runtime supports inter-process eBPF maps in user-space shared memory, as well as kernel eBPF maps, allowing seamless operation with the kernel's eBPF infrastructure. It also includes high-performance eBPF LLVM JIT/AOT compilers for multiple architectures.

Linux Security Modules (LSM) is a security framework implemented in the Linux kernel, providing a mechanism for various security policy modules to be inserted into the kernel, enhancing the system's security. LSM is designed to offer an abstraction layer for the Linux operating system to support multiple security policies without changing the core code of the kernel. This design allows system administrators or distributions to choose a security model that fits their security needs, such as SELinux, AppArmor, Smack, etc.

What can LSM be used for?

- Access Control: LSM is most commonly used to implement Mandatory Access Control (MAC) policies, different from the traditional owner-based Access Control (DAC). MAC can control access to resources like files, network ports, and inter-process communication in a fine-grained manner.
- Logging and Auditing: LSM can be used to log and audit sensitive operations on the system, providing detailed log information to help detect and prevent potential security threats.
- Sandboxing and Isolation: By limiting the behavior of programs and the resources they can access, LSM can sandbox applications, reducing the risk of malware or vulnerability exploitation.
- Enhancing Kernel and User-Space Security: LSM allows for additional security checks and restrictions to enhance the security of both the kernel itself and applications running in user-space.
- Limiting Privileged Operations: LSM can limit the operations that even processes with root privileges can perform, reducing the potential harm from misconfigurations by system administrators or malicious software with root access.

With bpftime, we can run eBPF programs in user space, compatible with the kernel, and collaborate with the kernel's eBPF to implement defense. Is it possible to further extend eBPF's security mechanisms and features to user space, allowing user-space eBPF and kernel-space eBPF to work together to implement more powerful and flexible security policies and defense capabilities? Let's call this mechanism USM (Userspace Security Modules or Union Security Modules).

You can explore more possibilities with us:

- Time Cost: ~350 hours
- Difficulty Level: Hard
- Mentors: Yiwei Yang (<yyang363@ucsc.edu>) Yusheng Zheng (<mailto:yunwei356@gmail.com>)

### Objectives

1. **USM Framework Design and Implementation**: Architect and implement the USM framework within bpftime, enabling user-space eBPF programs to work alongside kernel-space eBPF LSM programs.
2. **Security Scenario Exploration**: Investigate potential security scenarios where USM can effectively intercept and defend against security threats, using both kernel and user-space eBPF mechanisms.
3. **Continuous Integration and Testing**: Integrate USM testing into the bpftime CI pipeline, conducting regular checks to ensure compatibility and effectiveness of security policies.
4. **Documentation and Community Feedback**: Generate comprehensive documentation on USM's architecture, API, and implementation. Engage with the bpftime community to gather feedback and refine USM.
5. **Security Policy Development and Validation**: Develop and validate security policies that leverage USM, demonstrating its potential in enhancing system security.

### Expected Outcomes

- A fully implemented USM framework within the bpftime environment, allowing for seamless operation with kernel-space eBPF LSM programs and compatible with kernel eBPF toolchains and libraries.
- Integration of USM testing into the bpftime CI pipeline to ensure ongoing compatibility and security efficacy.
- A set of validated security policies showcasing USM's capability to enhance both kernel and user-space security.
- Comprehensive documentation and a feedback loop with the community for continuous improvement of USM.

### Prerequisites and Skills

- Proficiency in C/C++ and system programming.
- Understanding of security mechanisms and policies, especially related to Linux Security Modules (LSM) and eBPF.
- Familiarity with user-space and kernel-space programming paradigms.
- Experience with developing and testing eBPF programs is highly advantageous.

### Reference and Issue

- Conceptual foundation for USM in bpftime: [GitHub Discussion](https://github.com/eunomia-bpf/bpftime/issues/148)
- Initial exploration of eBPF security mechanisms: <https://docs.kernel.org/bpf/prog_lsm.html>, and kernel Runtime Verification <https://docs.kernel.org/trace/rv/runtime-verification.html#runtime-monitors-and-reactors>
- Engaging with existing eBPF and LSM communities for insights and collaboration opportunities.

## Add Fuzzer and kernel eBPF test for bpftime to improve compatibility

### Project Overview

The `bpftime` project, known for its innovative userspace eBPF runtime, is seeking to enhance its robustness and reliability by integrating a fuzzer. This project aims to develop or integrate a fuzzer for `bpftime`, using tools like [Google's Buzzer](https://github.com/google/buzzer) or `syzkaller`. The fuzzer will systematically test `bpftime` to uncover any potential bugs, memory leaks, or vulnerabilities, ensuring a more secure and stable runtime environment. Besides, we also need to add kernel eBPF test for bpftime to improve compatibility.

You also needs to enable the fuzzer and eBPF tests in CI.

### Timeframe and Difficulty

- **Time Commitment**: ~90 hours
- **Difficulty Level**: Easy

### Mentors

- Tong Yu ([yt.xyxx@gmail.com](mailto:yt.xyxx@gmail.com))
- Yusheng Zheng ([yunwei356@gmail.com](mailto:yunwei356@gmail.com))

### Objectives

1. **Fuzzer Development and Integration**: Design or develop a fuzzer that can be seamlessly integrated with `bpftime`. Or you can use existing fuzzers for eBPF.
2. **Testing and Debugging**: Use the fuzzer to identify and report bugs, memory leaks, or vulnerabilities in `bpftime` userspace eBPF runtime.
3. **Continuous Integration**: Integrate the fuzzer and kernel eBPF test into the `bpftime` CI pipeline, ensuring that it is run regularly to identify and resolve any issues.
4. **Documentation**: Create documentation detailing the fuzzer’s implementation or usage within the `bpftime` environment.
5. **Feedback Implementation**: Actively incorporate feedback from the `bpftime` community to refine and enhance the fuzzer.

### Expected Outcomes

- A fully integrated fuzzer within the `bpftime` environment.
- An integration of the fuzzer and kernel eBPF test into the `bpftime` CI pipeline.
- An increase in the identified and resolved bugs and vulnerabilities in `bpftime`.
- Documentation and guidelines for future contributors to utilize and improve the fuzzer.

### Prerequisites and Skills

- Skills in C/C++ and system programming.
- Familiarity with software testing methodologies, particularly fuzz testing.
- Experience with fuzzers like Google's Buzzer is highly beneficial.
- Basic knowledge of eBPF and its ecosystem.

### Reference and Issue

- Initial discussion on the need for a fuzzer in `bpftime`: [GitHub Issue](https://github.com/eunomia-bpf/bpftime/issues/163)
- Google buzzer: <https://github.com/google/buzzer>
- [FEATURE] Test with kernel eBPF test: <https://github.com/eunomia-bpf/bpftime/issues/210>

## Living patching distributed RocksDB with shared IO and Network Interface over io_uring

RocksDB is a high-performance, embedded key-value store for fast storage. It is widely used in distributed systems, such as databases, storage systems, and other applications. However, the performance of RocksDB is highly dependent on the underlying storage and network interfaces. The performance of RocksDB can be further improved by using shared IO and network interfaces over io_uring. This project aims to develop a living patching mechanism for distributed RocksDB with shared IO and network interfaces over io_uring, enabling dynamic and efficient performance optimization. This project will empower RocksDB with remote I/O and network interfaces, allowing it to leverage the performance benefits of io_uring and shared interfaces.

### Project Overview

- Time Cost: ~350 hours
- Difficulty Level: Hard
- Mentors: Yiwei Yang (<yyang363@ucsc.edu>) and Yusheng Zheng (<mailto:yunwei356@gmail.com>)

### Objectives

- Develop a living patching mechanism for distributed RocksDB with shared IO and network interfaces over io_uring.
- Implement a dynamic performance optimization system for distributed RocksDB, leveraging the performance benefits of io_uring MMAP interface.

### Expected Outcomes

- A living patching mechanism for distributed RocksDB with shared IO and network interfaces over io_uring.
- A dynamic performance optimization system for distributed RocksDB, leveraging the performance benefits of io_uring MMAP interface.

### Prerequisites and Skills

- Proficiency in C/C++ and system programming.
- Understanding of RocksDB and io_uring implementation.
- Familiarity with user-space and kernel-space programming paradigms.
- Experience with developing and testing eBPF programs is highly advantageous.

### Reference and Issue

- Recent paper about BPF function offloading to remote [BPF oF](https://arxiv.org/abs/2312.06808)
- eBPF meets io_uring [io_uring](https://lwn.net/Articles/847951/)

## Userspace AOT Compilation of eBPF for Lightweight Containers

### Overview

In the evolving world of cloud-native applications, IoT, and embedded systems, there's an increasing demand for efficient, secure, and resource-conscious computing solutions. Our project addresses these needs by focusing on the development of a userspace eBPF (Extended Berkeley Packet Filter) with Ahead-of-Time (AOT) compilation. This initiative aims to create a lightweight, event-driven computing model that caters to the unique demands of embedded and resource-constrained environments.

The main difference eBPF AOT can bring is that it can help build a verifiable and secure runtime for applications, and it can be lightweight and efficient enough, with a low startup time to run on embedded devices.

Duration and Difficulty Level

- Estimated Time: ~175 hours
- Difficulty Level: Medium
- Mentors: Tong Yu (<yt.xyxx@gmail.com>) Yusheng Zheng (<mailto:yunwei356@gmail.com>)

bpftime already have a AOT compiler, we need more work to enable it run on embedded devices or as plugins. If you want to add map support for microcontrollers with AOT compiler, maybe you can write a c implementation, compile it and link it with bpftime AOT products.

### Goals and Objectives

1. **Develop Userspace eBPF AOT Compilation**: The AOT compiler should be able to work well with helpers, ufuncs maps and other features of eBPF. Currently there is a POC for AOT compiler, but it's not complete and need more work.

You can choose one or two of these goals to work on:

1. **Integration into FaaS Containers**: Seamlessly integrate this technology into Function-as-a-Service (FaaS) lightweight containers, enhancing startup speed and operational efficiency.
2. **Plugin System Implementation**: Design a system allowing eBPF programs to be embedded as plugins in other applications, offering dynamic, optimized functionality.
3. **Run AOT eBPF on embedded devices**: Enable AOT eBPF to run on embedded devices, such as Raspberry Pi, and other IoT devices.

### Prerequisites and Skills Required

- Skills in C/C++ and system-level programming.
- Basic understanding of container technologies and FaaS architectures.
- Familiarity with eBPF concepts and applications.
- Interest in IoT, cloud-native, and embedded systems.

### Expected Outcomes

- A functional userspace eBPF runtime with AOT compilation capabilities.
- Demonstrated integration in FaaS lightweight containers.
- A plugin system enabling the embedding of eBPF programs in various applications.
- Run AOT eBPF on embedded devices.

### Additional Resources

1. The AOT example of bpftime
<https://github.com/eunomia-bpf/bpftime/blob/master/.github/workflows/test-aot-cli.yml>
2. The API for vm. <https://github.com/eunomia-bpf/bpftime/tree/master/vm/include>
3. Compile it as a standalone lib
<https://github.com/eunomia-bpf/bpftime/tree/master/vm/llvm-jit>
4. Femto-containers: lightweight virtualization and fault isolation for small software functions on low-power IoT microcontrollers <https://dl.acm.org/doi/abs/10.1145/3528535.3565242>

If you want to add map support for microcontrollers,  I think you can write a c implementation, compile it and link it with bpftime AOT products. We will provide an example later.

## Userspace eBPF for Userspace File System

In modern operating systems, `fuse` (Filesystem in Userspace) has become a popular choice, allowing developers to create file systems in user space without modifying kernel code. However, the cost of system calls still exists. This is where `bpftime` can play a role.

bpftime may help:

- reducing the overhead of system calls and enhancing performance
- enable cache for fuse without modifying the kernel
- dynamic adjustment of file system strategies based on performance data
- Add more policy and strategy for fuse

You can explore more possibilities with us:

- Time Cost: ~350 hours
- Difficulty Level: Hard
- Mentors: Yiwei Yang (<yyang363@ucsc.edu>) Yusheng Zheng (<mailto:yunwei356@gmail.com>)

### Objectives

1. **Synergistic Optimization Between User and Kernel Space**: Utilize `bpftime` to pre-process file system operations like caching and metadata queries in user space, thereby minimizing system call overhead.
2. **Kernel Bypass Mechanism in User Space**: Develop a kernel bypass mechanism for file systems in user space using eBPF, potentially eliminating the need for invasive changes to user applications.
3. **Dynamic Strategy Adjustment**: Implement a system within `bpftime` to dynamically collect performance data and adjust operational strategies in real-time.
4. **Customization for Specific Workloads**: Enable developers to tailor eBPF programs for diverse application scenarios, optimizing for various workloads.

### Expected Outcomes

- A proof-of-concept implementation demonstrating the synergy between `bpftime` and userspace file systems.
- A reduction in system call overhead for file operations in user space.
- A framework allowing dynamic adjustment of file system strategies based on performance data.
- Documentation or papers

### Prerequisites and Skills

- Proficiency in C/C++ and system-level programming.
- Familiarity with file system concepts and user space-kernel space interactions.
- Basic understanding of eBPF and its applications in modern operating systems.
- Experience with `fuse` or similar technologies is a plus.

### Resources

- Extfuse paper and GitHub repo: <https://github.com/extfuse/extfuse>
- <https://lwn.net/Articles/915717/>

## BPFTime Profiling and Machine Learning Prediction for far memory or distributed shared memory management

The upcoming world for CXL.mem provides a new way of memory fabric, it can seemingly share the memory between different nodes adding another layer between NUMA Remote, and SSDs. It can either be far memory node for disaggregation or distributed shared memory shared or pooled across nodes. However, issuing load and store to the CXL pool is easily throttle the performance. BPFTime can provide an extra layer of metrics collection and prediction for profiling guided memory management. BPFTime provides a cross kernel space and userspace boundary observability online. We think the offline access to the far memory is not deterministic across different workloads, and the same workloads with different runs, and the machine learning model can provide a better prediction for the memory access pattern.

### Project Overview

- Time Cost: ~350 hours
- Difficulty Level: Hard
- Mentors: Yiwei Yang (<yyang363@ucsc.edu>) Yusheng Zheng (<mailto:yunwei356@gmail.com>)

### Objectives

- Implement application specific metrics collection and profiling in BPFTime.
- Write eBPF for the far memory or distributed shared memory management.

### Expected Outcomes

- A set of metrics that can provide the right information for the memory scheduling and the memory access pattern.
- A set of eBPF programs that can provide the right metrics for the large language model Training or Inference.

### Prerequisites and Skills

- Proficiency in C/C++ and system programming.
- Understanding of kernel memory subsystem and memory management.
- Familiarity with user-space and kernel-space programming paradigms.
- Experience with developing and testing eBPF programs is highly advantageous.

### Reference and Issue

- eBPF for profiling: [eBPF for profiling](https://www.groundcover.com/ebpf/ebpf-profiling), eBPF for CPU scheduling: [eBPF for CPU scheduling](https://research.google/pubs/ghost-fast-and-flexible-user-space-delegation-of-linux-scheduling/)
- Paper's about ML for memory management in kernel: [Predicting Dynamic Properties of Heap Allocations](https://dl.acm.org/doi/pdf/10.1145/3591195.3595275) and [Towards a Machine Learning-Assisted Kernel with LAKE](https://dl.acm.org/doi/pdf/10.1145/3575693.3575697)
- State of the art far memory allocation [Pond](https://arxiv.org/abs/2203.00241), [Memtis](https://dl.acm.org/doi/10.1145/3600006.3613167), [MIRA](https://cseweb.ucsd.edu/~yiying/Mira-SOSP23.pdf) and [TMTS](https://www.micahlerner.com/assets/pdf/adaptable.pdf)

## Large Language Model specific metrics observability in BPFTime

BPFTime is able to provide multiple source of metrics in the userspace from the classical uprobe with maps. We can also provide metrics from gathering from the GPU, memory watch point, and other hardware. To support gdb rwatch BPFTime, we need to set a segfault to the certain memory accessed. For the GPU uprobe, we need static compilation and runtime API hooks to hook the certain GPU function calls. The uprobe attatched to the certain function calls provides the right online spot for annotate and make adjustment to the kernel's memory scheduling. The memory watch points can provide the memory access pattern and the memory access frequency. The GPU metrics can provide the GPU utilization and the memory access pattern. The combination of these metrics can provide the right information for the memory scheduling and the memory access pattern.

### Project Overview

- Time Cost: ~350 hours
- Difficulty Level: Hard
- Mentors: Yiwei Yang (<yyang363@ucsc.edu>) Yusheng Zheng (<mailto:yunwei356@gmail.com>)

### Objectives

- Provide the right metrics for the large language model Training or Inference.
- Programme the eBPF program to collect the right metrics and do the right scheduling.

### Expected Outcomes

- Implement the gdb rwatch and GPU metrics in BPFTime.
- A set of metrics that can provide the right information for the memory scheduling and the memory access pattern.
- A set of eBPF programs that can provide the right metrics for the large language model Training or Inference.

### Prerequisites and Skills

- Proficiency in C/C++ and system programming.
- Understanding the architecture of the large language model, and the metrics that are important for the performance.
- Has strong knowledge of GPU metrics collection, and gdb, perf, and other tools for metrics collection.
- Experience with developing and testing eBPF programs is highly advantageous.

### Reference and Issue

- Conceptual attach types discussion and in bpftime: [GitHub Discussion](https://github.com/eunomia-bpf/bpftime/issues/202)
- Papers about GPU metrics collection: [GPU metrics collection](https://itu-dasyalab.github.io/RAD/publication/papers/euromlsys2023.pdf) and [GPU static compilation and runtime API hooks](https://github.com/vosen/ZLUDA/blob/master/ARCHITECTURE.md#zluda-dumper)
- GDB's rwatch: [GDB rwatch](https://sourceware.org/gdb/onlinedocs/gdb/Set-Watchpoints.html) implemented on [X86](https://en.wikipedia.org/wiki/X86_debug_register) and [Arm](https://developer.arm.com/documentation/ka001494/latest/)

## Porting bpftime to Windows, FreeBSD, or other platforms

It would be similar to the porting to macOS.
