# Possible ideas for the future

This is some possible ideas for open source events, like GSOC(Google Summer of Code) or OSPP(Open Source Promotion Plan). Our projects are designed to suit contributors with varying levels of expertise, from students to more advanced developers.

It's also part of our project roadmap, if you don't participate in these events, you can also help or colaborate with these ideas! Need help? Please contact in the [email list](mailto:team@eunomia.dev) or in the [Discord channel](https://discord.gg/jvM73AFdB8).

## Table of contents

- [Possible ideas for the future](#possible-ideas-for-the-future)
  - [Table of contents](#table-of-contents)
  - [bpftime](#bpftime)
  - [Porting bpftime to macOS](#porting-bpftime-to-macos)
    - [Objectives for enable  eBPF on macOS](#objectives-for-enable--ebpf-on-macos)
    - [Expected Outcomes](#expected-outcomes)
    - [Prerequisites and Skills](#prerequisites-and-skills)
    - [Reference and issue](#reference-and-issue)
  - [Userspace AOT Compilation of eBPF for Lightweight Containers](#userspace-aot-compilation-of-ebpf-for-lightweight-containers)
    - [Overview](#overview)
    - [Goals and Objectives](#goals-and-objectives)
    - [Prerequisites and Skills Required](#prerequisites-and-skills-required)
    - [Expected Outcomes](#expected-outcomes-1)
    - [Additional Resources](#additional-resources)
  - [Add Fuzzer for bpftime and improve compatibility](#add-fuzzer-for-bpftime-and-improve-compatibility)
    - [Project Overview](#project-overview)
    - [Timeframe and Difficulty](#timeframe-and-difficulty)
    - [Mentors](#mentors)
    - [Objectives](#objectives)
    - [Expected Outcomes](#expected-outcomes-2)
    - [Prerequisites and Skills](#prerequisites-and-skills-1)
    - [Reference and Issue](#reference-and-issue-1)
  - [bpftime + fuse: Userspace eBPF for Userspace File System](#bpftime--fuse-userspace-ebpf-for-userspace-file-system)
    - [Objectives](#objectives-1)
    - [Expected Outcomes](#expected-outcomes-3)
    - [Prerequisites and Skills](#prerequisites-and-skills-2)
    - [Resources](#resources)

## bpftime

An userspace eBPF runtime that allows existing eBPF applications to operate in unprivileged userspace using the same libraries and toolchains. It offers Uprobe and Syscall tracepoints for eBPF, with significant performance improvements over kernel uprobe and without requiring manual code instrumentation or process restarts. The runtime facilitates interprocess eBPF maps in userspace shared memory, and is also compatible with kernel eBPF maps, allowing for seamless operation with the kernel's eBPF infrastructure. It includes a high-performance LLVM JIT for various architectures, alongside a lightweight JIT for x86 and an interpreter.

For more details, see:

- <https://eunomia.dev/bpftime>
- [https://github.com/eunomia-bpf/bpftime](https://github.com/eunomia-bpf/bpftime)

## Porting bpftime to macOS

Since bpftime can run in userspace and does not require kernel eBPF, why not enable eBPF on MacOS?

The goal of this project is to port `bpftime` to macOS, expanding its cross-platform capabilities and enabling macOS users to leverage the powerful features of `eBPF` in their development and production environments. With bpftime, now you may be able to run bcc and bpftrace tools on macOS!

- time: ~175 hour
- Difficulty Level: medium
- mentor: Yusheng Zheng (<yunwei356@gmail.com>) and Yuxi Huang (<Yuxi4096@gmail.com>)

### Objectives for enable  eBPF on macOS

1. **Compatibility and Integration**: Achieve compatibility of `bpftime` with macOS, ensuring that core features and capabilities are functional on this platform.
2. **Performance Optimization**: Fine-tune the performance of `bpftime` on macOS, focusing on optimizing the LLVM JIT and the lightweight JIT for x86 specifically for macOS architecture.
3. **Seamless Integration with macOS Ecosystem**: Ensure that `bpftime` integrates smoothly with macOS  environments, providing a native and efficient development experience for macOS eBPF users.
4. **Documentation and Tutorials**: Develop documentation and tutorials tailored to macOS users, facilitating easy adoption and use of `bpftime` on this platform.

### Expected Outcomes

- A functional port of `bpftime` for macOS, with core features operational.
- You should be able to run bpftrace and bcc tools on MacOS.
- documentation and guides for using `bpftime` on macOS.

### Prerequisites and Skills

- Proficiency in C/C++ and system programming.
- Familiarity with macOS development environment and tools.
- Understanding of eBPF and its applications.

### Reference and issue

- Issue and some initial discussion: <https://github.com/eunomia-bpf/bpftime/issues>
- Some previous efforts: [Enable bpftime on arm](https://github.com/eunomia-bpf/bpftime/pull/151)

## Userspace AOT Compilation of eBPF for Lightweight Containers

### Overview

In the evolving world of cloud-native applications, IoT, and embedded systems, there's an increasing demand for efficient, secure, and resource-conscious computing solutions. Our project addresses these needs by focusing on the development of a userspace eBPF (Extended Berkeley Packet Filter) with Ahead-of-Time (AOT) compilation. This initiative aims to create a lightweight, event-driven computing model that caters to the unique demands of embedded and resource-constrained environments.

The main difference eBPF AOT can bring is that it can help build a verifiable and secure runtime for applications, and it can be lightweight and efficient enough, with a low startup time to run on embedded devices.

Duration and Difficulty Level

- Estimated Time: ~175 hours
- Difficulty Level: Medium
- Mentors: Tong Yu (<yt.xyxx@gmail.com>) Yusheng Zheng (<yunwei356@gmail.com>)

bpftime already have a AOT compiler, we need more work to enable it run on embedded devices or as plugins.

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

If you want to add map support for microcontrollers,  I think you can write a c implementation, compile it and link it with bpftime AOT products. We will provide an example later.

## Add Fuzzer for bpftime and improve compatibility

### Project Overview

The `bpftime` project, known for its innovative userspace eBPF runtime, is seeking to enhance its robustness and reliability by integrating a fuzzer. This project aims to develop and integrate a fuzzer specifically designed for `bpftime`, using tools like [Google's Buzzer](https://github.com/google/buzzer). The fuzzer will systematically test `bpftime` to uncover any potential bugs, memory leaks, or vulnerabilities, ensuring a more secure and stable runtime environment.

You also needs to enable the fuzzer in CI.

### Timeframe and Difficulty

- **Time Commitment**: ~90 hours
- **Difficulty Level**: Easy

### Mentors

- Tong Yu ([yt.xyxx@gmail.com](mailto:yt.xyxx@gmail.com))
- Yusheng Zheng ([yunwei356@gmail.com](mailto:yunwei356@gmail.com))

### Objectives

1. **Fuzzer Development and Integration**: Design and develop a fuzzer that can be seamlessly integrated with `bpftime`. Or you can use existing fuzzers for eBPF.
2. **Testing and Debugging**: Use the fuzzer to identify and report bugs, memory leaks, or vulnerabilities in `bpftime` userspace eBPF runtime.
3. **Documentation**: Create documentation detailing the fuzzer’s implementation and usage within the `bpftime` environment.
4. **Feedback Implementation**: Actively incorporate feedback from the `bpftime` community to refine and enhance the fuzzer.

### Expected Outcomes

- A fully integrated fuzzer within the `bpftime` environment.
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

## bpftime + fuse: Userspace eBPF for Userspace File System

In modern operating systems, `fuse` (Filesystem in Userspace) has become a popular choice, allowing developers to create file systems in user space without modifying kernel code. However, the cost of system calls still exists. This is where `bpftime` can play a role.

bpftime may help:

- reducing the overhead of system calls and enhancing performance
- enable cache for fuse without modifying the kernel
- dynamic adjustment of file system strategies based on performance data
- Add more policy and strategy for fuse

You can explore more possibilities with us:

- Time Cost: ~350 hours
- Difficulty Level: Hard
- Mentors: Yiwei Yang (<yyang363@ucsc.edu>) Yusheng Zheng (<yunwei356@gmail.com>)

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
