# Examples & Use Cases

## Table of Contents

- [Examples \& Use Cases](#examples--use-cases)
  - [Table of Contents](#table-of-contents)
  - [minimal examples](#minimal-examples)
  - [Tracing the system](#tracing-the-system)
  - [Error injections](#error-injections)
  - [Nginx eBPF module](#nginx-ebpf-module)

## minimal examples

See [example/minimal](https://github.com/eunomia-bpf/bpftime/tree/master/example/minimal).

The bpftime supports the following types of eBPF programs:

- `uprobe/uretprobe`: trace userspace functions at start or and.
- `syscall tracepoints`: trace the specific syscall types.

You may use `bpf_override_return` to change the control flow of the program.

See [documents/available-features.md](https://github.com/eunomia-bpf/bpftime/tree/master/documents/avaliable-features.md) for more details.

## Tracing the system

- `tracing userspace functions with uprobe`: Attach uprobe, uretprobe or all syscall tracepoints(currently x86 only) eBPF programs to a process or a group of processes:
  - [`malloc`](https://github.com/eunomia-bpf/bpftime/tree/master/example/malloc): count the malloc calls in libc by pid. demonstrate how to use the userspace `uprobe` with basic `hashmap`.
  - [`bashreadline`](https://github.com/eunomia-bpf/bpftime/tree/master/example/libbpf-tools/bashreadline): Print entered bash commands from running shells,
  - [`sslsniff`](https://github.com/eunomia-bpf/bpftime/tree/master/example/sslsniff): Trace and print all SSL/TLS connections and raw traffic data.
- `tracing all syscalls with tracepoints`
  - [`opensnoop`](https://github.com/eunomia-bpf/bpftime/tree/master/example/opensnoop): trace file open or close syscalls in a process. demonstrate how to use the userspace `syscall tracepoint` with `ring buffer` output.

More bcc/libbpf-tools examples can be found in [example/libbpf-tools](https://github.com/eunomia-bpf/bpftime/tree/master/example/libbpf-tools).

You can also run bpftime with `bpftrace`, we've test it on [this commit](https://github.com/iovisor/bpftrace/commit/75aca47dd8e1d642ff31c9d3ce330e0c616e5b96). More details about how to run bpftrace in usespace, can be found in [example/bpftrace](https://github.com/eunomia-bpf/bpftime/tree/master/example/bpftrace).

## Error injections

- [`error-injection`](https://github.com/eunomia-bpf/bpftime/tree/master/example/error-inject) Inject errors into userspace functions or syscalls to test its error handling capabilities.

## Nginx eBPF module

A nginx eBPF module is implemented with bpftime, which can be used to extend nginx with eBPF programs.

See https://github.com/eunomia-bpf/Nginx-eBPF-module
