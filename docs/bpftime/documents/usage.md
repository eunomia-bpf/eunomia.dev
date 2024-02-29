# Manual

🚧 It's at an early stage and may contain bugs on more platforms and eBPF programs. We are working on to improve the stability and compatibility. It's not suitable for production use now.

If you find any bugs or suggestions, please feel free to open an issue, thanks!

## Table of Contents

- [Manual](#manual)
  - [Table of Contents](#table-of-contents)
  - [Uprobe and uretprobe](#uprobe-and-uretprobe)
  - [Syscall tracing](#syscall-tracing)
  - [Run with LD\_PRELOAD directly](#run-with-ld_preload-directly)
  - [Run with JIT enabled](#run-with-jit-enabled)
  - [Run with kernel eBPF](#run-with-kernel-ebpf)
  - [Control Log Level](#control-log-level)

## Uprobe and uretprobe

Uprobes(User Probe) allow you to attach dynamic probes to user-space functions, enabling you to monitor and trace specific user-level functions or methods in your applications. bpftime can be utilized within BPF programs attached to uprobes to capture timestamps for different events related to that function.

With `bpftime`, you can build eBPF applications using familiar tools like clang and libbpf, and execute them in userspace. For instance, the `malloc` eBPF program traces malloc calls using uprobe and aggregates the counts using a hash map.

You can refer to [documents/build-and-test.md](build-and-test.md) for how to build the project.

To get started, you can build and run a libbpf based eBPF program starts with `bpftime` cli:

```console
make -C example/malloc # Build the eBPF program example
bpftime load ./example/malloc/malloc
```

In another shell, Run the target program with eBPF inside:

```console
$ bpftime start ./example/malloc/victim
Hello malloc!
malloc called from pid 250215
continue malloc...
malloc called from pid 250215
```

You can also dynamically attach the eBPF program with a running process:

```console
$ ./example/malloc/victim & echo $! # The pid is 101771
[1] 101771
101771
continue malloc...
continue malloc...
```

And attach to it:

```console
$ sudo bpftime attach 101771 # You may need to run make install in root
Inject: "/root/.bpftime/libbpftime-agent.so"
Successfully injected. ID: 1
```

You can see the output from original program:

```console
$ bpftime load ./example/malloc/malloc
...
12:44:35 
        pid=247299      malloc calls: 10
        pid=247322      malloc calls: 10
```

Alternatively, you can also run our sample eBPF program directly in the kernel eBPF, to see the similar output:

```console
$ sudo example/malloc/malloc
15:38:05
        pid=30415       malloc calls: 1079
        pid=30393       malloc calls: 203
        pid=29882       malloc calls: 1076
        pid=34809       malloc calls: 8
```

## Syscall tracing

Syscall tracing involves monitoring system calls made by processes running on your system. By adding bpftime to your BPF programs, you can capture timestamps for various syscall events, such as entry and exit.

An example can be found at [examples/opensnoop](https://github.com/eunomia-bpf/bpftime/tree/master/example/opensnoop)

```console
$ sudo ~/.bpftime/bpftime load ./example/opensnoop/opensnoop
[2023-10-09 04:36:33.891] [info] manager constructed
[2023-10-09 04:36:33.892] [info] global_shm_open_type 0 for bpftime_maps_shm
[2023-10-09 04:36:33][info][23999] Enabling helper groups ffi, kernel, shm_map by default
PID    COMM              FD ERR PATH
72101  victim             3   0 test.txt
72101  victim             3   0 test.txt
72101  victim             3   0 test.txt
72101  victim             3   0 test.txt
```

In another terminal, run the victim program:

```console
$ sudo ~/.bpftime/bpftime start -s example/opensnoop/victim
[2023-10-09 04:38:16.196] [info] Entering new main..
[2023-10-09 04:38:16.197] [info] Using agent /root/.bpftime/libbpftime-agent.so
[2023-10-09 04:38:16.198] [info] Page zero setted up..
[2023-10-09 04:38:16.198] [info] Rewriting executable segments..
[2023-10-09 04:38:19.260] [info] Loading dynamic library..
...
test.txt closed
Opening test.txt
test.txt opened, fd=3
Closing test.txt...
```

## Run with LD_PRELOAD directly

If the command line interface is not enough, you can also run the eBPF program with `LD_PRELOAD` directly.

The command line tool is a wrapper of `LD_PRELOAD` and can work with `ptrace` to inject the runtime shared library into a running target process.

Run the eBPF tool with libbpf:

```sh
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/malloc/malloc
```

Start the target program to trace:

```sh
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so example/malloc/victim
```

## Run with JIT enabled

If the performance is not good enough, you can try to enable JIT. The JIT will be enabled by default in the future after more tests.

Set `BPFTIME_USE_JIT=true` in the server to enable JIT, for example, when running the server:

```sh
LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so BPFTIME_USE_JIT=true example/malloc/malloc
```

The default behavior is using ubpf JIT, you can also use LLVM JIT by compile with LLVM JIT enabled. See [documents/build-and-test.md](build-and-test.md) for more details.

## Run with kernel eBPF

You can run the eBPF program in userspace with kernel eBPF in two ways. The kernel must have eBPF support enabled, and kernel version should be higher enough to support mmap eBPF map.

1. with the shared library `libbpftime-syscall-server.so`, for example:

```sh
BPFTIME_NOT_LOAD_PATTERN=start_.* BPFTIME_RUN_WITH_KERNEL=true LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so example/malloc/malloc
```

2. Using daemon mode, see <https://github.com/eunomia-bpf/bpftime/tree/master/daemon>

## Control Log Level

Set `SPDLOG_LEVEL` to control the log level dynamically, for example, when running the server:

```sh
SPDLOG_LEVEL=debug LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so example/malloc/malloc
```

Available log level include:

- trace
- debug
- info
- warn
- err
- critical
- off

See <https://github.com/gabime/spdlog/blob/v1.x/include/spdlog/cfg/env.h> for more details.

Log can also be controled at compile time by specifying `-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO` in the cmake compile command.
