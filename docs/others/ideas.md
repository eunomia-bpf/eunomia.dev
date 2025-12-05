# Possible ideas for the future

This is some possible ideas for open source events, like GSOC(Google Summer of Code) or OSPP(Open Source Promotion Plan) and others. Our projects are designed to suit contributors with varying levels of expertise, from students to more advanced developers.

It's also part of our project roadmap, if you don't participate in these events, you can also help or colaborate with these ideas! Need help? Please contact in the [email list](mailto:team@eunomia.dev) or in the [Discord channel](https://discord.gg/jvM73AFdB8).

- <https://eunomia.dev/bpftime>
- [https://github.com/eunomia-bpf/bpftime](https://github.com/eunomia-bpf/bpftime)

## Persistent eBPF Maps with Crash‑Consistent Semantics

### Project Overview

Kernel eBPF maps are frequently treated as small in‑kernel KV stores, but today they provide **no persistence or crash‑consistency semantics**. Pinned maps survive loader exit but not reboots; updates have no durability guarantees; and applications build fragile ad‑hoc recovery logic.
This project aims to design and prototype **persistent eBPF maps** with explicit, documented semantics for **durability, crash recovery, and versioning**—suitable for production observability, policy modules, and long‑running agents.

* Time Cost: ~200 hours
* Difficulty Level: Hard
* Mentors: Yusheng Zheng yunwei356@gmail.com

### Objectives

* Define a persistence model for kernel maps, including durability levels, epoch barriers, permitted partial states after crashes, and expected reconstruction rules.
* Build a persistence backend (privileged user‑space daemon + kernel API usage) that mirrors map mutations into an append‑only log and periodic snapshots, with recovery logic that reconstructs kernel maps on reboot.
* Integrate persistence without modifying verifier or map types initially; persistence is additive through metadata and a daemon, with an optional future “persistent map” flag.
* Implement automatic recovery at boot: detect persistent maps, rebuild state, repopulate kernel maps before programs attach, and require no application‑specific handlers.

### Expected Outcomes

* A documented durability and crash‑consistency model for eBPF maps.
* A daemon and on‑disk format for logging and snapshotting kernel map state.
* A crash‑injection test suite validating correctness of recovery.
* Example cases such as persistent counters, node‑local policies, and replicated state.

May have a chance to publish papers on top conference.

### Useful References

* [Running BPF After Application Exits (eunomia blog)](https://eunomia.dev/tutorials/bpf-application-exits/)
* [kernel.org – eBPF Map Documentation](https://docs.kernel.org/bpf/map_data.html)

## sched_ext‑Based Coz‑Style Causal Profiler (“SchedCoz”)

### Project Overview

Coz ([paper](https://web.mit.edu/PLV/papers/coz-sosp15.pdf)) introduced **causal profiling** via *virtual speedups*: when a target line executes, other threads are artificially slowed to approximate the effect of optimizing that line. Coz implements this in user space with `LD_PRELOAD` and injected sleeps.
This project re‑implements causal profiling **inside the kernel scheduler** using **sched_ext**, producing a cleaner, zero‑injection mechanism suitable for real multi‑process services and container environments.

* Time Cost: ~150 hours
* Difficulty Level: Hard
* Mentors: Yusheng Zheng yunwei356@gmail.com

### Objectives

* Use eBPF perf‑event sampling and uprobes/USDT to collect user stacks and progress points; a user‑space controller selects regions and coordinates experiments.
* Implement a sched_ext scheduler that maintains a global “debt” incremented when the target region executes, and reduces CPU service to tasks with unpaid debt, reproducing Coz’s virtual‑speedup semantics.
* Establish correctness by mapping Coz’s delay‑insertion model to scheduler‑level service‑reduction, handling multi‑core behavior, preemption boundaries, and wakeup dependencies.
* Provide scoped deployment by confining experiments to specific cgroups and ensuring clean fallback to CFS.

### Expected Outcomes

* A functioning **SchedCoz** profiler that integrates eBPF sampling and sched_ext scheduling to perform causal experiments.
* Experimental fidelity comparable to Coz, with potentially lower variance and no user‑space sleeps or preload libraries.
* Demonstrations on multi‑process workloads (Memcached, Redis, SQLite, NGINX) and containerized environments.
* Documentation and runnable examples for building sched_ext‑based profilers.

May have a chance to publish papers on top conference.


### Useful References

* [Coz: Finding Code that Counts (SOSP’15)](https://web.mit.edu/PLV/papers/coz-sosp15.pdf)
* [sched_ext Kernel Documentation](https://docs.kernel.org/scheduler/sched-ext.html)
* [BPF perf_event Programs](https://docs.kernel.org/bpf/bpf_perf_event.html)


## Porting bpftime to macOS and Windows, FreeBSD, or other platforms

Since bpftime can run in userspace and does not require kernel eBPF, why not enable eBPF on MacOS/FreeBSD/Other Platforms?

The goal of this project is to port `bpftime` to macOS and other platforms, expanding its cross-platform capabilities and enabling macOS users to leverage the powerful features of `eBPF` in their development and production environments. With bpftime, now you may be able to run bcc and bpftrace tools on macOS and other OSs!

- time: ~175 hour
- Difficulty Level: medium
- mentor: Tong Yu (<yt.xyxx@gmail.com>) and Yuxi Huang (<Yuxi4096@gmail.com>)

### Objectives for enable eBPF on macOS and Windows, FreeBSD

1. **Compatibility and Integration**: Achieve compatibility of `bpftime` with macOS and/or other OSs, ensuring that core features and capabilities are functional on this platform.
2. **Performance Optimization**: Fine-tune the performance of `bpftime` on macOS and/or other OSs, focusing on optimizing the LLVM JIT and the lightweight JIT for x86 specifically for macOS architecture.
3. **Seamless Integration with macOS Ecosystem**: Ensure that `bpftime` integrates smoothly with macOS and/or other OSs environments, providing a native and efficient development experience for eBPF users.
4. **Documentation and Tutorials**: Develop documentation and tutorials tailored to macOS users, facilitating easy adoption and use of `bpftime` on this platform.

### Expected Outcomes

- A functional port of `bpftime` for macOS  and Windows, FreeBSD, with core features operational.
- You should be able to run `bpftrace` and `bcc` tools on them, and get expected output.
- documentation and guides for using `bpftime` on macOS and/or other OSs.

### Prerequisites and Skills

- Proficiency in C/C++ and system programming.
- Familiarity with macOS development environment and tools.
- Understanding of eBPF and its applications.

### Reference and issue

- Issue and some initial discussion: <https://github.com/eunomia-bpf/bpftime/issues>
- Some previous efforts: [Enable bpftime on arm](https://github.com/eunomia-bpf/bpftime/pull/151)

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

## BPFTime Profiling and Machine Learning Prediction for far memory or distributed shared memory management

The upcoming world for CXL.mem provides a new way of memory fabric, it can seemingly share the memory between different nodes adding another layer between NUMA Remote, and SSDs. It can either be far memory node for disaggregation or distributed shared memory shared or pooled across nodes. However, issuing load and store to the CXL pool is easily throttle the performance. BPFTime can provide an extra layer of metrics collection and prediction for profiling guided memory management. BPFTime provides a cross kernel space and userspace boundary observability online. We think the offline access to the far memory is not deterministic across different workloads, and the same workloads with different runs, and the machine learning model can provide a better prediction for the memory access pattern.

### Project Overview

- Time Cost: ~350 hours
- Difficulty Level: Hard
- Mentors:  Tong Yu (<yt.xyxx@gmail.com>) and Yusheng Zheng (<mailto:yunwei356@gmail.com>)

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
- Mentors:  Tong Yu (<yt.xyxx@gmail.com>) and Yusheng Zheng (<mailto:yunwei356@gmail.com>)

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
<!-- ## APX-aware JIT backend for legacy x86 and bpftime

Modern Intel CPUs with APX (Advanced Performance Extensions) expose 32 general-purpose registers and richer 3-operand encodings, offering significant potential for reducing spills and memory traffic in hot code paths. Many existing binaries, JITs, and runtimes, however, still emit “legacy” x86-64 code that cannot automatically take advantage of APX. This project aims to build an APX-aware JIT / dynamic binary translation backend that can “rehydrate” legacy x86-64 code into an intermediate representation (IR) and re-emit it using APX features for maximum performance when running on APX-capable CPUs.

This JIT can be integrated with bpftime’s userspace runtime (e.g., for helpers, ufuncs, and host-side instrumentation code), or used as a standalone component for accelerating hot regions of existing x86-64 applications.

- Time Cost: ~350 hours  
- Difficulty Level: Hard  
- Mentors: Yiwei Yang (<yyang363@ucsc.edu>), Yusheng Zheng (<mailto:yunwei356@gmail.com>)

### Project Overview

The goal of this project is to design and implement an APX-aware JIT backend that:

- Detects APX support on the host CPU.
- Lifts legacy x86-64 code (or bpftime-generated code) into an SSA-like IR.
- Performs APX-specific optimizations (extra registers, 3-operand forms, flag suppression).
- Emits APX machine code into a code cache and transparently routes hot paths through the APX-optimized version.

For bpftime, this enables a next-generation userspace runtime where both eBPF programs and their surrounding helper logic can benefit from APX when available, while still falling back to standard x86-64 on older hardware.

### Objectives

1. **APX-capable CPU Detection and Dispatch**
   - Implement runtime feature detection for APX-capable processors.
   - Provide a clean CPU dispatch layer that selects APX or legacy code paths at startup or JIT time.

2. **IR Lifting from Legacy x86-64**
   - Decode legacy x86-64 basic blocks or traces into an intermediate representation (SSA-like).
   - Model registers, flags, and memory accesses so that APX-specific optimizations can be applied cleanly.
   - Integrate this IR with bpftime’s existing VM / LLVM JIT infrastructure where appropriate.

3. **APX-specific Optimization Passes**
   - Use APX’s extra general-purpose registers (R16–R31) to eliminate spills and stack traffic in hot blocks.
   - Convert classic 2-operand arithmetic into 3-operand APX forms to shorten dependency chains.
   - Use flag-suppression forms (where available) when flags are not needed, reducing EFLAGS pressure.
   - Explore small patterns where conditional loads/stores can replace short branches.

4. **Code Generation and Code Cache Management**
   - Implement an APX-aware register allocator that prefers EGPRs (R16–R31) for short-lived temporaries.
   - Emit APX-encoded instructions into a code cache and manage patching/jump trampolines from original code.
   - Provide safe fallbacks for unsupported or self-modifying code (e.g., interpretation or legacy re-emission).

5. **Integration with bpftime and Tooling**
   - Expose the APX JIT backend as an optional path for bpftime’s userspace runtime (e.g., for helpers, ufuncs, and hot loops in host code).
   - Add configuration switches and environment variables to enable/disable APX optimizations.
   - Provide benchmarks showing improvements for representative bpftime workloads (profiling, networking, file-system helpers, etc.).

6. **Documentation and Evaluation**
   - Document the design of the IR, the APX-specific passes, and integration points with bpftime.
   - Provide microbenchmarks (e.g., arithmetic kernels, memcopy-like loops) and macrobenchmarks (bpftime-based tools) comparing:
     - Legacy x86-64 code,
     - APX-aware JIT code.
   - Summarize trade-offs in code size, JIT overhead, and runtime performance.

### Expected Outcomes

- A working APX-aware JIT backend capable of:
  - Lifting legacy x86-64 code into an IR.
  - Re-emitting it using APX extensions on supported CPUs.
- Integration hooks for bpftime so that bpftime’s userspace runtime can optionally use APX-optimized code for hot paths.
- Benchmarks and evaluation showing tangible performance benefits (e.g., reduced spills, better throughput) on APX hardware.
- Clear documentation and examples demonstrating how to enable APX JIT, how it behaves on non-APX CPUs, and how contributors can extend or customize the optimization passes.

### Prerequisites and Skills

- Strong C/C++ and systems programming skills.
- Familiarity with x86-64 architecture, instruction encoding, and CPU microarchitecture concepts.
- Experience with JIT compilation, dynamic binary translation, or compiler IR (e.g., LLVM IR, custom SSA).
- Basic understanding of eBPF, bpftime’s goals, and userspace runtime design is highly desirable.
- Experience with performance profiling and benchmarking on modern CPUs.

### Reference and Issue

- Intel APX and extended GPRs / instruction encodings (official whitepapers and manuals).
- Existing JIT/DBT frameworks and code caches (e.g., LLVM ORC JIT, QEMU TCG, DynamoRIO, etc.).
- bpftime runtime and VM code:
  - <https://eunomia.dev/bpftime>  
  - <https://github.com/eunomia-bpf/bpftime>
- A future GitHub issue in `eunomia-bpf/bpftime` can be created to track design, discussion, and implementation progress for the APX JIT backend. -->
