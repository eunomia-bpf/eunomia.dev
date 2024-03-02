# How bpftime works?

bpftime consists of two main components: a syscall-compatible library and an attachment agent. The syscall-compatible library interfaces with userspace eBPF applications, such as bcc-tools or bpftrace, and translates eBPF-related system calls into function calls. The library also creates shared memory segments for eBPF programs and maps, which are used for communication and data exchange between the userspace eBPF runtime and the control plane application. The attachment agent is a shared library that can be dynamically injected into the target process using ptrace or LD_PRELOAD. The agent is responsible for hooking the target functions or syscalls using binary rewriting techniques, and redirecting the execution flow to the userspace eBPF programs. The userspace eBPF programs can then access the host environment by updating userspace maps, using userspace helpers, or invoking userspace Foreign Function Interface (FFI) functions.


## How the bpftime work entirely in userspace


- **Binary rewriting**:$~$ bpftime modifies the binary code of the target process at runtime to insert breakpoints and trampolines for uprobe and syscall hooks. It uses a syscall server and agent to communicate with the target process and perform the rewriting safely and transparently.
- **Uprobe hooks**:$~$ bpftime inserts a breakpoint instruction at the entry point of the target function, which triggers a SIGTRAP signal when executed. The signal handler then transfers the execution to the eBPF program, which can access and modify the registers and stack of the target process. After the eBPF program finishes, the execution returns to the original function via a trampoline.
- **Syscall hooks**:$~$ bpftime replaces the syscall instruction with a breakpoint instruction, which also triggers a SIGTRAP signal. The signal handler then checks the syscall number and decides whether to run the eBPF program or not. The eBPF program can filter, modify, or redirect the syscall arguments and return value. After the eBPF program finishes, the execution resumes with the original or modified syscall.
- **eBPF maps**:$~$ bpftime implements eBPF maps in shared memory, which can be accessed by multiple processes and eBPF programs. It supports various types of maps, such as hash, array, stack, queue, etc. It also supports kernel eBPF maps, which can be used to cooperate with kernel eBPF programs.




### The bpftime runtime can work with kernel eBPF in two ways:

- **Loading userspace eBPF from kernel**:$~$ bpftime can load eBPF bytecode from the kernel using the `bpf` system call, and execute it in userspace. This allows userspace eBPF programs to access kernel eBPF maps and cooperate with kernel eBPF programs, such as kprobes and network filters.
- **Using kernel eBPF maps**:$~$ bpftime can use the `bpf` system call to create or attach to kernel eBPF maps, and use them in userspace eBPF programs. This enables userspace eBPF programs to share data with kernel eBPF programs, and leverage the existing kernel eBPF infrastructure.

For more details, you can refer to the [GitHub repository](https://github.com/eunomia-bpf/bpftime) or the [blog post](https://arxiv.org/abs/2311.07923) of bpftime.

## References:

(1) GitHub - eunomia-bpf/bpftime: Userspace eBPF runtime for fast Uprobe .... https://github.com/eunomia-bpf/bpftime\
(2) bpftime: Extending eBPF from Kernel to User Space - Medium. https://medium.com/@yunwei356/bpftime-extending-ebpf-from-kernel-to-user-space-5cafea3a5a98\
(3) bpftime: Extending eBPF from Kernel to User Space - eunomia. https://eunomia.dev/blogs/bpftime/\
(4) Userspace eBPF Runtimes: Overview and Applications - Medium. https://medium.com/@yunwei356/userspace-ebpf-runtimes-overview-and-applications-d19e3c84c7a7\
(5) bpftime: userspace eBPF Runtime for Uprobe, Syscall and Kernel-User .... https://arxiv.org/pdf/2311.07923.pdf\
(6) bpftime: userspace eBPF Runtime for Uprobe, Syscall and Kernel-User .... https://ar5iv.labs.arxiv.org/html/2311.07923\
(7) bpftime: userspace eBPF Runtime for Uprobe, Syscall and Kernel-User .... https://arxiv.org/abs/2311.07923