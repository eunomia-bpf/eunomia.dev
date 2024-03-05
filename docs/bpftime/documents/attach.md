# Detailed Implementation of the Attach Part

## A Brief Description

Attach is an important part of bpftime.

`Attach` refers to connecting eBPF programs stored in memory to some userspace hook point. Such hook points could be a userspace function in a certain process, a syscall invocation, or something similar. When the hook point is triggered, the connected eBPF program is called automatically. The whole process is managed by bpftime.

bpftime itself comes with two types of attach implementations: uprobe and syscall.
- uprobe: Hooks a userspace function in a certain process, monitoring the arguments and return value, or modifying the return value based on conditions.
- syscall: Hooks calls to some or all syscall IDs in a certain process. It monitors arguments or return values of the syscall, or replaces the syscall implementation.

The attach implementations above are similar to `uprobe` and `tracepoint/syscalls/sys_[enter/exit]_XXXX` in the kernel, with some enhancements.

## Targets (Libraries) in bpftime Related to Attach

### base_attach_impl

This is a header-only library, providing abstractions for all attach implementations. There are two major classes: `base_attach_impl` and `attach_private_data`.

`base_attach_impl` is the base class for all attach implementations. It is composed of several pure virtual functions, including functions to detach a certain attach entry and create an attach entry with a unified interface.

`attach_private_data` is the base class for all attach private data. Since we can only access a certain attach implementation through the unified interface, we also need a unified interface to pass attach-specific data to the attach implementation. Such data are called `attach private data`

### frida_uprobe_attach_impl

A Frida-based uprobe attach implementation. It implements the following attach types:
- UPROBE: Similar to `uprobe` in kernel eBPF, invoked at the entrance of a certain userspace function. It can access the arguments of the hooked function.
- URETPROBE: Similar to `uretprobe` in kernel eBPF, invoked at the exit of a certain userspace function. It can access the return value of the hooked function.

UPROBE and URETPROBE are implemented using `GUM_INVOCATION_LISTENER`.

- UPROBE_OVERRIDE: Invoked when the function enters, it can access arguments of the hooked function, and decide whether to override the return value and bypass the hooked function. It could be used to replace the hooked function or replace the return value based on arguments.

UFILTER are implemented using `gum_interceptor_replace`

### syscall_trace_attach_impl

A zpoline-based userspace syscall trace implementation. It rewrites the code segment to trace syscalls in userspace, eliminating the need to modify the executable.

It provides the attach type SYSCALL_TRACEPOINT, which is triggered when the userspace process emits a syscall. It can monitor the arguments, invoke the original syscall, and replace the syscall.

It is similar to `tracepoint/syscalls/sys_[enter/exit]_XXXX` in the kernel eBPF.

### Runtime

This is the core part of bpftime and an important part of attaching.

User applications, such as bpftime-agent, should register attach implementations to `bpf_attach_ctx`, a class managed by runtime. In this way, runtime doesn't need to depend on concrete attach implementations; it just needs to depend on `base_attach_impl`, which provides a general interface for all attach implementations.

When the user application wants to instantiate an attach, which means:
- Creating an eBPF virtual machine based on eBPF programs stored in memory.
- Creating attach private data through the unified interface from perf events stored in memory.
- Creating an attach entry through the unified interface.

runtime will iterate over all handlers stored in memory and handle different types. The type related to attach is `bpf_link_handler`, which records the relationship between `bpf_prog_handler` and `perf_event_handler`, meaning when the specified perf event is triggered, the corresponding eBPF program should be called.

runtime will call `create_attach_with_ebpf_callback` to create an attach entry. This function is provided by `base_attach_impl` and implemented by various attach implementations. runtime will generate the attach private data using the registered attach private data generator and create a lambda callback to call the corresponding eBPF program. After attaching, runtime will record the attach entry ID for future purposes (such as detachment).
