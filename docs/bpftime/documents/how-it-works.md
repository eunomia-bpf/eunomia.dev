# The design and implementation of bpftime

The hook implementation is based on binary rewriting and the underly technique is inspired by:

- Userspace function hook: [frida-gum](https://github.com/frida/frida-gum)
- Syscall hooks: [zpoline: a system call hook mechanism based on binary rewriting](https://www.usenix.org/conference/atc23/presentation/yasukata) and [pmem/syscall_intercept](https://github.com/pmem/syscall_intercept).

For more details about how to implement the inline hook, please refer to our blog: [Implementing an Inline Hook in C in 5 minutes](../../blogs/inline-hook.md) and the demo <https://github.com/eunomia-bpf/inline-hook-demo>

The injection of userspace eBPF runtime into a running program is based on ptrace and also provided by [frida-gum](https://github.com/frida/frida-gum) library.

### How the bpftime work entirely in userspace:

![How it works](bpftime.png)

### How the bpftime work with kernel eBPF:

![How it works with kernel eBPF](bpftime-kernel.png)

For more details, please refer to:

- Slides: <https://eunomia.dev/bpftime/documents/userspace-ebpf-bpftime-lpc.pdf>
- arxiv: <https://arxiv.org/abs/2311.07923>
