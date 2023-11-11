# eBPF安全性的新篇章：面临的挑战与前沿创新

郑昱笙

扩展伯克利数据包过滤器（eBPF）代表了我们与现代操作系统交互和扩展其能力方式的重大演变。作为一种强大的技术，它使得Linux内核能够响应事件运行沙盒程序，eBPF已成为系统可观察性、网络和安全特性的基石。

然而，像任何与内核紧密接口的系统一样，eBPF本身的安全性至关重要。在这篇博客中，我们将深入探讨常被忽视的eBPF安全性问题，探索旨在保护eBPF的机制本身如何被加固。我们将解析eBPF验证器的作用，审视当前的访问控制模型，并调查持续研究中的潜在改进。此外，我们将通过eBPF的安全复杂性，解决系统架构师和开发者面临的开放性问题和挑战。

## 目录

<!-- TOC -->

- [eBPF安全性的新篇章：面临的挑战与前沿创新](#ebpf安全性的新篇章面临的挑战与前沿创新)
  - [目录](#目录)
  - [如何通过验证器确保 eBPF 的安全](#如何通过验证器确保-ebpf-的安全)
    - [Challenges](#challenges)
    - [Other works to improve verifier](#other-works-to-improve-verifier)
  - [Limitations in eBPF Access Control](#limitations-in-ebpf-access-control)
    - [CAP\_BPF](#cap_bpf)
    - [bpf namespace](#bpf-namespace)
    - [Unprivileged eBPF](#unprivileged-ebpf)
      - [Trusted Unprivileged BPF](#trusted-unprivileged-bpf)
  - [Other possible solutions](#other-possible-solutions)
  - [Conclusion](#conclusion)

<!-- /TOC -->

## 如何通过验证器确保 eBPF 的安全

- **遵循控制流程图**
  验证器首先通过构建并遵循eBPF程序的控制流程图（CFG）来进行其分析。它细致地计算出每条指令的所有可能状态，同时考虑BPF寄存器集和堆栈。然后根据当前的指令上下文进行安全检查。

  其中一个关键步骤是跟踪程序私有BPF堆栈的寄存器溢出/填充情况。这确保了堆栈相关操作不会引起溢出或下溢，避免了数据破坏或成为攻击路径。

- **控制流程图的回边处理**
  验证器通过识别CFG中的回边来有效处理eBPF程序内的循环。通过模拟所有迭代直到达到预定的上限，从而确保循环不会导致无限制执行。

- **处理大量潜在状态**
  验证器需要处理程序执行路径中大量潜在状态带来的复杂性。它运用路径修剪逻辑，将当前状态与之前的状态进行比较，判断当前路径是否与之前的路径“

等效”，并且有一个安全的出口。这样减少了需要考虑的状态总数。

- **逐函数验证以减少状态数量**
  为了简化验证过程，验证器进行逐函数分析。这种模块化的方法使得在任何给定时间内需要分析的状态数量得以减少，从而提高了验证过程的效率。

- **按需标量精度追踪以进一步减少状态**
  验证器运用按需标量精度追踪来进一步减少状态空间。通过在必要时对标量值进行回溯，验证器可以更准确地预测程序的行为，优化其分析过程。

- **超过“复杂性”阈值时终止并拒绝**
  为了保持实用性能，验证器设定了一个“复杂性”阈值。如果程序分析超过此阈值，验证器将终止过程并拒绝该程序。这样确保只有在可管理的复杂性范围内的程序被允许执行，实现了安全性与系统性能的平衡。

### Challenges

Despite its thoroughness, the eBPF verifier faces significant challenges:

- **Attractive target for exploitation when exposed to non-root users**
  As the verifier becomes more complex, it becomes an increasingly attractive target for exploitation. The programmability of eBPF, while powerful, also means that if an attacker were to bypass the verifier and gain execution within the OS kernel, the consequences could be severe.

- **Reasoning about verifier correctness is non-trivial**
  Ensuring the verifier's correctness, especially concerning Spectre mitigations, is not a straightforward task. While there is some formal verification in place, it is only partial. Areas such as the Just-In-Time (JIT) compilers and abstract interpretation models are particularly challenging.

- **Occasions where valid programs get rejected**
  There is sometimes a disconnect between the optimizations performed by LLVM (the compiler infrastructure used to prepare eBPF programs) and the verifier's ability to understand these optimizations, leading to valid programs being erroneously rejected.

- **"Stable ABI" for BPF program types**
  A "stable ABI" is vital so that BPF programs running in production do not break upon an OS kernel upgrade. However, maintaining this stability while also evolving the verifier and the BPF ecosystem presents its own set of challenges.

- **Performance vs. security considerations**
  Finally, the eternal trade-off between performance and security is pronounced in the verification of complex eBPF programs. While the verifier must be efficient to be practical, it also must not compromise on security, as the performance of the programs it is verifying is crucial for modern computing systems.

The eBPF verifier stands as a testament to the ingenuity in modern computing security, navigating the treacherous waters between maximum programmability and maintaining a fortress-like defense at the kernel level.

### Other works to improve verifier

- Specification and verification in the field: Applying formal methods to BPF just-in-time compilers in the Linux kernel: <https://www.usenix.org/conference/osdi20/presentation/nelson>
- "Sound, Precise, and Fast Abstract Interpretation with Tristate Numbers”, Vishwanathan et al. <https://arxiv.org/abs/2105.05398>
- “Eliminating bugs in BPF JITs using automated formal verification”, Nelson et al. <https://arxiv.org/abs/2105.05398>
- “A proof-carrying approach to building correct and flexible BPF verifiers”, Nelson et al. <https://linuxplumbersconf.org/event/7/contributions/685/>
- “Automatically optimizing BPF programs using program synthesis”, Xu et al. <https://linuxplumbersconf.org/event/11/contributions/944/>
- “Simple and Precise Static Analysis of Untrusted Linux Kernel Extensions”, Gershuni et al. <https://linuxplumbersconf.org/event/11/contributions/951/>
- “An Analysis of Speculative Type Confusion Vulnerabilities in the Wild”, Kirzner et al. <https://www.usenix.org/conference/usenixsecurity21/presentation/kirzner>

Together, these works signify a robust and multi-faceted research initiative aimed at bolstering the foundations of eBPF verification, ensuring that it remains a secure and performant tool for extending the capabilities of the Linux kernel.

Other reference for you to learn more about eBPF verifier:

- BPF and Spectre: Mitigating transient execution attacks: <https://popl22.sigplan.org/details/prisc-2022-papers/11/BPF-and-Spectre-Mitigating-transient-execution-attacks>

## Limitations in eBPF Access Control

After leading Linux distributions, such as Ubuntu and SUSE, have disallowed unprivileged usage of eBPF Socket Filter and CGroup programs, the current eBPF access control model only supports a single permission level. This level necessitates the CAP_SYS_ADMIN capability for all features. However, CAP_SYS_ADMIN carries inherent risks, particularly to containers, due to its extensive privileges.

Addressing this, Linux 5.6 introduces a more granular permission system by breaking down eBPF capabilities. Instead of relying solely on CAP_SYS_ADMIN, a new capability, CAP_BPF, is introduced for invoking the bpf syscall. Additionally, installing specific types of eBPF programs demands further capabilities, such as CAP_PERFMON for performance monitoring or CAP_NET_ADMIN for network administration tasks. This structure aims to mitigate certain types of attacks—like altering process memory or eBPF maps—that still require CAP_SYS_ADMIN.

Nevertheless, these segregated capabilities are not bulletproof against all eBPF-based attacks, such as Denial of Service (DoS) and information theft. Attackers may exploit these to craft eBPF-based malware specifically targeting containers. The emergence of eBPF in cloud-native applications exacerbates this threat, as users could inadvertently deploy containers that contain untrusted eBPF programs.

Compounding the issue, the risks associated with eBPF in containerized environments are not entirely understood. Some container services might unintentionally grant eBPF permissions, for reasons such as enabling filesystem mounting functionality. The existing permission model is inadequate in preventing misuse of these potentially harmful eBPF features within containers.

### CAP_BPF

Traditionally, almost all BPF actions required CAP_SYS_ADMIN privileges, which also grant broad system access. Over time, there has been a push to separate BPF permissions from these root privileges. As a result, capabilities like CAP_PERFMON and CAP_BPF were introduced to allow more granular control over BPF operations, such as reading kernel memory and loading tracing or networking programs, without needing full system admin rights.

However, CAP_BPF's scope is ambiguous, leading to a perception problem. Unlike CAP_SYS_MODULE, which is well-defined and used for loading kernel modules, CAP_BPF lacks namespace constraints, meaning it can access all kernel memory rather than being container-specific. This broad access is problematic because verifier bugs in BPF programs can crash the kernel, considered a security vulnerability, leading to an excessive number of CVEs (Common Vulnerabilities and Exposures) being filed, even for bugs that are already fixed. This response to verifier bugs creates undue alarm and urgency to patch older kernel versions that may not have been updated.

Additionally, some security startups have been criticized for exploiting the fears around BPF's capabilities to market their products, paradoxically using BPF itself to safeguard against the issues they highlight. This has led to a contradictory narrative where BPF is both demonized and promoted as a solution.

### bpf namespace

The current security model requires the CAP_SYS_ADMIN capability for iterating BPF object IDs and converting these IDs to file descriptors (FDs). This is to prevent non-privileged users from accessing BPF programs owned by others, but it also restricts them from inspecting their own BPF objects, posing a challenge in container environments.

Users can run BPF programs with CAP_BPF and other specific capabilities, yet they lack a generic method to inspect these programs, as tools like bpftool need CAP_SYS_ADMIN. The existing workaround without CAP_SYS_ADMIN is deemed inconvenient, involving SCM_RIGHTS and Unix domain sockets for sharing BPF object FDs between processes.

To address these limitations, Yafang Shao proposes introducing a BPF namespace. This would allow users to create BPF maps, programs, and links within a specific namespace, isolating these objects from users in different namespaces. However, objects within a BPF namespace would still be visible to the parent namespace, enabling system administrators to maintain oversight.

The BPF namespace is conceptually similar to the PID namespace and is intended to be intuitive. The initial implementation focuses on BPF maps, programs, and links, with plans to extend this to other BPF objects like BTF and bpffs in the future. This could potentially enable container users to trace only the processes within their container without accessing data from other containers, enhancing security and usability in containerized environments.

reference:

- BPF and security: <https://lwn.net/Articles/946389/>
- Cross Container Attacks: The Bewildered eBPF on Clouds <https://www.usenix.org/system/files/usenixsecurity23-he.pdf>
- bpf: Introduce BPF namespace: <https://lwn.net/Articles/927354/>
- ebpf-running-in-linux-namespaces: <https://stackoverflow.com/questions/48815633/ebpf-running-in-linux-namespaces>

### Unprivileged eBPF

The concept of unprivileged eBPF refers to the ability for non-root users to load eBPF programs into the kernel. This feature is controversial due to security implications and, as such, is currently turned off by default across all major Linux distributions. The concern stems from hardware vulnerabilities like Spectre to kernel bugs and exploits, which can be exploited by malicious eBPF programs to leak sensitive data or attack the system.

To combat this, mitigations have been put in place for various versions of these vulnerabilities, like v1, v2, and v4. However, these mitigations come at a cost, often significantly reducing the flexibility and performance of eBPF programs. This trade-off makes the feature unattractive and impractical for many users and use cases.

#### Trusted Unprivileged BPF

In light of these challenges, a middle ground known as "trusted unprivileged BPF" is being explored. This approach would involve an allowlist system, where specific eBPF programs that have been thoroughly vetted and deemed trustworthy could be loaded by unprivileged users. This vetting process would ensure that only secure, production-ready programs bypass the privilege requirement, maintaining a balance between security and functionality. It's a step toward enabling more widespread use of eBPF without compromising the system's integrity.

- Permissive LSM hooks: Rejected upstream given LSMs enforce further restrictions

    New Linux Security Module (LSM) hooks specifically for the BPF subsystem, with the intent of offering more granular control over BPF maps and BTF data objects. These are fundamental to the operation of modern BPF applications.

    The primary addition includes two LSM hooks: bpf_map_create_security and bpf_btf_load_security, which provide the ability to override the default permission checks that rely on capabilities like CAP_BPF and CAP_NET_ADMIN. This new mechanism allows for finer control, enabling policies to enforce restrictions or bypass checks for trusted applications, shifting the decision-making to custom LSM policy implementations.

    This approach allows for a safer default by not requiring applications to have BPF-related capabilities, which are typically required to interact with the kernel's BPF subsystem. Instead, applications can run without such privileges, with only vetted and trusted cases being granted permission to operate as if they had elevated capabilities.

- BPF token concept to delegate subset of BPF via token fd from trusted privileged daemon

    the BPF token, a new mechanism allowing privileged daemons to delegate a subset of BPF functionality to trusted unprivileged applications. This concept enables containerized BPF applications to operate safely within user namespaces—a feature previously unattainable due to security restrictions with CAP_BPF capabilities. The BPF token is created and managed via kernel APIs, and it can be pinned within the BPF filesystem for controlled access. The latest version of the patch ensures that a BPF token is confined to its creation instance in the BPF filesystem to prevent misuse. This addition to the BPF subsystem facilitates more secure and flexible unprivileged BPF operations.

- BPF signing as gatekeeper: application vs BPF program (no one-size-fits-all)

    Song Liu has proposed a patch for unprivileged access to BPF functionality through a new device, `/dev/bpf`. This device controls access via two new ioctl commands that allow users with write permissions to the device to invoke `sys_bpf()`. These commands toggle the ability of the current task to call `sys_bpf()`, with the permission state being stored in the `task_struct`. This permission is also inheritable by new threads created by the task. A new helper function, `bpf_capable()`, is introduced to check if a task has obtained permission through `/dev/bpf`. The patch includes updates to documentation and header files.

- RPC to privileged BPF daemon: Limitations depending on use cases/environment

    The RPC approach (eg. bpfd) is similar to the BPF token concept, but it uses a privileged daemon to manage the BPF programs. This daemon is responsible for loading and unloading BPF programs, as well as managing the BPF maps. The daemon is also responsible for verifying the BPF programs before loading them. This approach is more flexible than the BPF token concept, as it allows for more fine-grained control over the BPF programs. However, it is also more complex, bring more maintenance challenges and possibilities for single points of failure.

reference

- Permissive LSM hooks: <https://lore.kernel.org/bpf/20230412043300.360803-1-andrii@kernel.org/>
- BPF token concept: <https://lore.kernel.org/bpf/20230629051832.897119-1-andrii@kernel.org/>
- BPF signing using fsverity and LSM gatekeeper: <https://www.youtube.com/watch?v=9p4qviq60z8>
- Sign the BPF bytecode: <https://lpc.events/event/16/contributions/1357/attachments/1045/1999/BPF%20Signatures.pdf>
- bpfd: <https://bpfd.dev/>

## Other possible solutions

Some research or discussions about how to improve the security of eBPF. Existing works can be roughly divided into three categories: virtualization, Software Fault Isolation (SFI), and
formal methods.

- MOAT: Towards Safe BPF Kernel Extension (Isolation)

    The Linux kernel makes considerable use of
    Berkeley Packet Filter (BPF) to allow user-written BPF applications
    to execute in the kernel space. BPF employs a verifier to
    statically check the security of user-supplied BPF code. Recent
    attacks show that BPF programs can evade security checks and
    gain unauthorized access to kernel memory, indicating that the
    verification process is not flawless. In this paper, we present
    MOAT, a system that isolates potentially malicious BPF programs
    using Intel Memory Protection Keys (MPK). Enforcing BPF
    program isolation with MPK is not straightforward; MOAT is
    carefully designed to alleviate technical obstacles, such as limited
    hardware keys and supporting a wide variety of kernel BPF
    helper functions. We have implemented MOAT in a prototype
    kernel module, and our evaluation shows that MOAT delivers
    low-cost isolation of BPF programs under various real-world
    usage scenarios, such as the isolation of a packet-forwarding
    BPF program for the memcached database with an average
    throughput loss of 6%.

    <https://arxiv.org/abs/2301.13421>

    > If we must resort to hardware protection mechanisms, is language safety or verification still necessary to protect the kernel and extensions from one another?

- Unleashing Unprivileged eBPF Potential with Dynamic Sandboxing

    For safety reasons, unprivileged users today have only limited ways to customize the kernel through the extended Berkeley Packet Filter (eBPF). This is unfortunate, especially since the eBPF framework itself has seen an increase in scope over the years. We propose SandBPF, a software-based kernel isolation technique that dynamically sandboxes eBPF programs to allow unprivileged users to safely extend the kernel, unleashing eBPF's full potential. Our early proof-of-concept shows that SandBPF can effectively prevent exploits missed by eBPF's native safety mechanism (i.e., static verification) while incurring 0%-10% overhead on web server benchmarks.

    <https://arxiv.org/abs/2308.01983>

    > Is the original design of eBPF not to be a sandbox? Why not using webassembly for SFI?

- Kernel extension verification is untenable

    The emergence of verified eBPF bytecode is ushering in a
    new era of safe kernel extensions. In this paper, we argue
    that eBPF’s verifier—the source of its safety guarantees—has
    become a liability. In addition to the well-known bugs and
    vulnerabilities stemming from the complexity and ad hoc
    nature of the in-kernel verifier, we highlight a concerning
    trend in which escape hatches to unsafe kernel functions
    (in the form of helper functions) are being introduced to
    bypass verifier-imposed limitations on expressiveness, unfortunately also bypassing its safety guarantees. We propose
    safe kernel extension frameworks using a balance of not
    just static but also lightweight runtime techniques. We describe a design centered around kernel extensions in safe
    Rust that will eliminate the need of the in-kernel verifier,
    improve expressiveness, allow for reduced escape hatches,
    and ultimately improve the safety of kernel extensions

    <https://sigops.org/s/conferences/hotos/2023/papers/jia.pdf>

    > Is it limits the kernel to load only eBPF programs that are signed by trusted third parties, as the kernel itself can no longer independently verify them? The rust toolchains also has vulnerabilities?

## Conclusion

As we have traversed the multifaceted domain of eBPF security, it's clear that while eBPF’s verifier provides a robust first line of defense, there are inherent limitations within the current access control model that require attention. We have considered potential solutions from the realms of virtualization, software fault isolation, and formal methods, each offering unique approaches to fortify eBPF against vulnerabilities. However, as with any complex system, new questions and challenges continue to surface. The gaps identified between the theoretical security models and their practical implementation invite continued research and experimentation. The future of eBPF security is not only promising but also demands a collective effort to ensure the technology can be adopted with confidence in its capacity to safeguard systems
