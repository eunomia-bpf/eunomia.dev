# How does bpftime work? 

Two key components make up bpftime: a syscall-compatible library and an attachment agent. The library understands the eBPF-related system calls made by userspace eBPF applications (e.g., bcc-tools, bpftrace) and maps them into function calls. It helps in creating shared memory segments for eBPF programs and maps used for communication and data exchange between userspace eBPF runtime and control plane application. The agent is a shared library that can be loaded dynamically into the target process either through ptrace or `LD_PRELOAD`. Agent, on the other hand, uses binary rewriting techniques to hook target functions or syscalls then divert execution flow to userspace eBPF programs . Thereafter, these userspace eBPF programs will be able to access host environment by updating userspace maps, using userspace helpers or invoking Foreign Function Interface (FFI) functions in the user space 

## How does the bpftime work entirely in userspace 

- **Binary rewriting**: At runtime, this modifies binary code of target process so as to insert breakpoints as well as trampolines for uprobe and syscall hooks. It employs a syscall server and agent that interact with the target process for safe transparent rewriting.

- **Uprobe hooks**: bpftime inserts a breakpoint instruction into the target function’s entry point that will trigger a signal of type `SIGTRAP` when the program is run. Then, the execution transfers to an eBPF program in the signal handler that can access and edit registers and stacks of a targeted process. Finally, eBPF program finishes its work and returns to the initial function using trampoline.

- **Syscall hooks**: Signal trap will be triggered by SIGTRAP as soon as system calls gives way for it through the replacement of its instructions with break points. SYS_ number is verified then whether or not eBFP is launched using the signal handler. It is there that eBPF modifies return value along with parameter passing and arguments redirecting during system calls. When eBPF has finished execution, it resumes at original or modified system call.

- **eBPF maps**: For instance, bpftime provides shared memory implementation of eBPF maps, which can be accessed from multiple processes and programs based on eBPF map operations. Hashmap is one among other forms including arrays, stacks, queue etc. Also supported by it are kernelized eBPF maps for cooperating with kernelized eBPF programs.


### The bpftime runtime can work with kernel eBPF in two ways:

- **Kernel-Based User-Space eBPF Loading**: bpftime can load eBPF bytecode from the kernel by using the `bpf` system call and executing it in user space. This allows userspace eBPF programs to use the kernel’s BPF maps, working together with other kernel’s eBPF programs like network filters and kprobes.

- **Using Kernel eBPF Maps**: bpftime is able to use `bpf` system call in order to create or attach itself to a kernel eBPF map; then it can also consume such maps in its own userspace-based eBPF functions. By that, it ensures that userspace-based eBPF solutions can share data with their kernels’ counterparts, thus taking full advantage of existing infrastructure provided by the latter.

For more details, refer to the [GitHub repository](https://github.com/eunomia-bpf/bpftime) or the [blog post](https://arxiv.org/abs/2311.07923) of bpftime.

## References:

1. GitHub - eunomia-bpf/bpftime: Userspace eBPF runtime for fast Uprobe .... <https://github.com/eunomia-bpf/bpftime>
2. bpftime: Extending eBPF from Kernel to User Space - Medium. <https://medium.com/@yunwei356/bpftime-extending-ebpf-from-kernel-to-user-space-5cafea3a5a98>
3. bpftime: Extending eBPF from Kernel to User Space - eunomia. <https://eunomia.dev/blogs/bpftime/>
4. Userspace eBPF Runtimes: Overview and Applications - Medium. <https://medium.com/@yunwei356/userspace-ebpf-runtimes-overview-and-applications-d19e3c84c7a7>
5. bpftime: userspace eBPF Runtime for Uprobe, Syscall and Kernel-User .... <https://arxiv.org/pdf/2311.07923.pdf>
6. bpftime: userspace eBPF Runtime for Uprobe, Syscall and Kernel-User .... <https://ar5iv.labs.arxiv.org/html/2311.07923>
7. bpftime: userspace eBPF Runtime for Uprobe, Syscall and Kernel-User .... <https://arxiv.org/abs/2311.07923>