# The security in eBPF

In this blog, we will focus on the security in eBPF. We will not talk about how to use eBPF to improve the security of the system, but focus on the security of eBPF itself.

## Introduction

## How eBPF ensures security with verifier

The verifier is the key to ensure the security of eBPF.

### What the eBPF verifier is and what it does

Static code analyzer walking in-kernel copy of BPF program instructions

- Ensuring program termination
  - DFS traversal to check program is a DAG
  - Preventing unbounded loops
  - Preventing out-of-bounds or malformed jumps
- Ensuring memory safety
  - Preventing out-of-bounds memory access
  - Preventing use-after-free bugs and object leaks
  - Also mitigating vulnerabilities in the underlying hardware (Spectre)
- Ensuring type safety
  - Preventing type confusion bugs
  - BPF Type Format (BTF) for access to (kernel’s) aggregate types
- Preventing hardware exceptions (division by zero)
  - For unknown scalars, instructions rewritten to follow aarch64 spec

### How the eBPF verifier works

Works by simulating execution of all paths of the program

- Follows control flow graph
  - For each instruction computes set of possible states (BPF register set & stack)
  - Performs safety checks (e.g. memory access) depending on current instruction
  - Register spill/fill tracking for program’s private BPF stack
- Back-edges in control flow graph
  - Bounded loops by brute-force simulating all iterations up to a limit
- Dealing with potentially large number of states
  - Path pruning logic compares current state vs prior states
    - Current path “equivalent” to prior paths with safe exit?
- Function-by-function verification for state reduction
- On-demand scalar precision (back-)tracking for state reduction
- Terminates with rejection upon surpassing “complexity” threshold

### Challenges

- Attractive target for exploitation when exposed to non-root
  - Growing verifier complexity
  - Programmability can be abused to bypass mitigations once in OS kernel
- Reasoning about verifier correctness is non-trivial
  - Especially Spectre mitigations
  - Only partial formal verification (e.g. tnums, JITs)
- Occasions where valid programs get rejected
  - LLVM vs verifier “disconnect” to understand optimizations
  - Restrictions when tracking state
- “Stable ABI” for BPF program types (with some limitations)
  - BPF programs in production should not break upon OS kernel upgrade
- Performance vs security considerations
  - Verification of complex programs must be efficient to be practical
  - Mitigations must be practical as performance of programs crucial

### Previous works to improve security of verifier

- Specification and verification in the field: Applying formal methods to BPF just-in-time compilers in the Linux kernel: <https://www.usenix.org/conference/osdi20/presentation/nelson>
- "Sound, Precise, and Fast Abstract Interpretation with Tristate Numbers”, Vishwanathan et al. <https://arxiv.org/abs/2105.05398>
- “Eliminating bugs in BPF JITs using automated formal verification”, Nelson et al. <https://arxiv.org/abs/2105.05398>
- “A proof-carrying approach to building correct and flexible BPF verifiers”, Nelson et al. <https://linuxplumbersconf.org/event/7/contributions/685/>
- “Automatically optimizing BPF programs using program synthesis”, Xu et al. <https://linuxplumbersconf.org/event/11/contributions/944/>
- “Simple and Precise Static Analysis of Untrusted Linux Kernel Extensions”, Gershuni et al. <https://linuxplumbersconf.org/event/11/contributions/951/>
- “An Analysis of Speculative Type Confusion Vulnerabilities in the Wild”, Kirzner et al. <https://www.usenix.org/conference/usenixsecurity21/presentation/kirzner>

reference:

- BPF and Spectre: Mitigating transient execution attacks: <https://popl22.sigplan.org/details/prisc-2022-papers/11/BPF-and-Spectre-Mitigating-transient-execution-attacks>

## Unprivileged eBPF

Unprivileged BPF:

- Currently disabled by default in all major distributions
- Not possible any time soon due to CPU HW vulnerabilities (Spectre & co)
- Mitigations for v1/v2/v4 implemented but severely limit flexibility and performance

### trusted unprivileged BPF

Potential compromise is “trusted unprivileged BPF”, Allowlist exceptions for trusted and verified production use cases

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

- Unleashing Unprivileged eBPF Potential with Dynamic Sandboxing

    For safety reasons, unprivileged users today have only limited ways to customize the kernel through the extended Berkeley Packet Filter (eBPF). This is unfortunate, especially since the eBPF framework itself has seen an increase in scope over the years. We propose SandBPF, a software-based kernel isolation technique that dynamically sandboxes eBPF programs to allow unprivileged users to safely extend the kernel, unleashing eBPF's full potential. Our early proof-of-concept shows that SandBPF can effectively prevent exploits missed by eBPF's native safety mechanism (i.e., static verification) while incurring 0%-10% overhead on web server benchmarks.

    <https://arxiv.org/abs/2308.01983>

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
