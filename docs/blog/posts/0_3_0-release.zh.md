---
date: 2023-02-11
---

# eunomia-bpf 0.3.0 发布：只需编写内核态代码，轻松构建、打包、发布完整的 eBPF 应用

## eunomia-bpf 简介

eBPF 源于 BPF，本质上是处于内核中的一个高效与灵活的虚拟机组件，以一种安全的方式在许多内核 hook 点执行字节码，开发者可基于 eBPF 开发性能分析工具、软件定义网络、安全等诸多场景。但是，目前对于开发和使用 eBPF 应用而言还可能存在一些不够方便的地方：

- 搭建和开发 eBPF 程序是一个门槛比较高、比较复杂的工作，必须同时关注内核态和用户态两个方面的交互和信息处理，有时还要配置环境和编写对应的构建脚本；
- 目前不同用户态语言如 C、Go、Rust 等编写的工具难以兼容、难以统一管理，多种开发生态难以整合：如何跨架构、跨语言和内核版本，使用标准化的方式方便又快捷的打包、分发、发布二进制 eBPF 程序，同时还需要能很方便地动态调整 eBPF 程序的挂载点、参数等等？
- 如何更方便地使用 eBPF 的工具：有没有可能从云端一行命令拉下来就使用，类似 docker 那样？或者把 eBPF 程序作为服务运行，通过 HTTP 请求和 URL 即可热更新、动态插拔运行任意一个 eBPF 程序？

[eunomia-bpf](https://github.com/eunomia-bpf/eunomia-bpf) 是一个开源的 eBPF 动态加载运行时和开发工具链，是为了简化 eBPF 程序的开发、构建、分发、运行而设计的，基于 libbpf 的 CO-RE 轻量级开发框架。
<!-- more -->

使用 eunomia-bpf ，可以：

- 在编写 eBPF 程序或工具时只编写内核态代码，自动获取内核态导出信息；
- 使用 Wasm 进行用户态交互程序的开发，在 Wasm 虚拟机内部控制整个 eBPF 程序的加载和执行，以及处理相关数据；
- eunomia-bpf 可以将预编译的 eBPF 程序打包为通用的 JSON 或 Wasm 模块，跨架构和内核版本进行分发，无需重新编译即可动态加载运行。

eunomia-bpf 由一个编译工具链和一个运行时库组成, 对比传统的 BCC、原生 libbpf 等框架，大幅简化了 eBPF 程序的开发流程，在大多数时候只需编写内核态代码，即可轻松构建、打包、发布完整的 eBPF 应用，同时内核态 eBPF 代码保证和主流的 libbpf, libbpfgo, libbpf-rs 等开发框架的 100% 兼容性。需要编写用户态代码的时候，也可以借助 Webassembly(Wasm) 实现通过多种语言进行用户态开发。和 bpftrace 等脚本工具相比, eunomia-bpf 保留了类似的便捷性, 同时不仅局限于 trace 方面, 可以用于更多的场景, 如网络、安全等等。

> - eunomia-bpf 项目 Github 地址: <https://github.com/eunomia-bpf/eunomia-bpf>
> - gitee 镜像: <https://gitee.com/anolis/eunomia>

我们发布了最新的 0.3 版本, 对于整体的开发和使用流程进行了优化，同时也支持了更多的 eBPF 程序和 maps 类型。

## 运行时优化：增强功能性, 增加多种程序类型

1. 只需编写内核态代码, 即可获得对应的输出信息, 以可读、规整的方式打印到标准输出. 以一个简单的 eBPF 程序, 跟踪所有 open 类型系统调用的 opensnoop 为例:

    头文件 opensnoop.h

    ```c
    #ifndef __OPENSNOOP_H
    #define __OPENSNOOP_H

    #define TASK_COMM_LEN 16
    #define NAME_MAX 255
    #define INVALID_UID ((uid_t)-1)

    // used for export event
    struct event {
      /* user terminology for pid: */
      unsigned long long ts;
      int pid;
      int uid;
      int ret;
      int flags;
      char comm[TASK_COMM_LEN];
      char fname[NAME_MAX];
    };

    #endif /* __OPENSNOOP_H */
    ```

    内核态代码 opensnoop.bpf.c

    ```c
    #include <vmlinux.h>
    #include <bpf/bpf_helpers.h>
    #include "opensnoop.h"

    struct args_t {
      const char *fname;
      int flags;
    };

    /// Process ID to trace
    const volatile int pid_target = 0;
    /// Thread ID to trace
    const volatile int tgid_target = 0;
    /// @description User ID to trace
    const volatile int uid_target = 0;
    /// @cmdarg {"default": false, "short": "f", "long": "failed"}
    const volatile bool targ_failed = false;

    struct {
      __uint(type, BPF_MAP_TYPE_HASH);
      __uint(max_entries, 10240);
      __type(key, u32);
      __type(value, struct args_t);
    } start SEC(".maps");

    struct {
      __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
      __uint(key_size, sizeof(u32));
      __uint(value_size, sizeof(u32));
    } events SEC(".maps");

    static __always_inline bool valid_uid(uid_t uid) {
      return uid != INVALID_UID;
    }

    static __always_inline
    bool trace_allowed(u32 tgid, u32 pid)
    {
      u32 uid;

      /* filters */
      if (tgid_target && tgid_target != tgid)
        return false;
      if (pid_target && pid_target != pid)
        return false;
      if (valid_uid(uid_target)) {
        uid = (u32)bpf_get_current_uid_gid();
        if (uid_target != uid) {
          return false;
        }
      }
      return true;
    }

    SEC("tracepoint/syscalls/sys_enter_open")
    int tracepoint__syscalls__sys_enter_open(struct trace_event_raw_sys_enter* ctx)
    {
      u64 id = bpf_get_current_pid_tgid();
      /* use kernel terminology here for tgid/pid: */
      u32 tgid = id >> 32;
      u32 pid = id;

      /* store arg info for later lookup */
      if (trace_allowed(tgid, pid)) {
        struct args_t args = {};
        args.fname = (const char *)ctx->args[0];
        args.flags = (int)ctx->args[1];
        bpf_map_update_elem(&start, &pid, &args, 0);
      }
      return 0;
    }

    SEC("tracepoint/syscalls/sys_enter_openat")
    int tracepoint__syscalls__sys_enter_openat(struct trace_event_raw_sys_enter* ctx)
    {
      u64 id = bpf_get_current_pid_tgid();
      /* use kernel terminology here for tgid/pid: */
      u32 tgid = id >> 32;
      u32 pid = id;

      /* store arg info for later lookup */
      if (trace_allowed(tgid, pid)) {
        struct args_t args = {};
        args.fname = (const char *)ctx->args[1];
        args.flags = (int)ctx->args[2];
        bpf_map_update_elem(&start, &pid, &args, 0);
      }
      return 0;
    }

    static __always_inline
    int trace_exit(struct trace_event_raw_sys_exit* ctx)
    {
      struct event event = {};
      struct args_t *ap;
      int ret;
      u32 pid = bpf_get_current_pid_tgid();

      ap = bpf_map_lookup_elem(&start, &pid);
      if (!ap)
        return 0; /* missed entry */
      ret = ctx->ret;
      if (targ_failed && ret >= 0)
        goto cleanup; /* want failed only */

      /* event data */
      event.pid = bpf_get_current_pid_tgid() >> 32;
      event.uid = bpf_get_current_uid_gid();
      bpf_get_current_comm(&event.comm, sizeof(event.comm));
      bpf_probe_read_user_str(&event.fname, sizeof(event.fname), ap->fname);
      event.flags = ap->flags;
      event.ret = ret;

      /* emit event */
      bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU,
                &event, sizeof(event));

    cleanup:
      bpf_map_delete_elem(&start, &pid);
      return 0;
    }

    SEC("tracepoint/syscalls/sys_exit_open")
    int tracepoint__syscalls__sys_exit_open(struct trace_event_raw_sys_exit* ctx)
    {
      return trace_exit(ctx);
    }

    SEC("tracepoint/syscalls/sys_exit_openat")
    int tracepoint__syscalls__sys_exit_openat(struct trace_event_raw_sys_exit* ctx)
    {
      return trace_exit(ctx);
    }

    /// Trace open family syscalls.
    char LICENSE[] SEC("license") = "GPL";
    ```

    编译运行:

    ```console
    $ ecc opensnoop.bpf.c opensnoop.h
    Compiling bpf object...
    Generating export types...
    Packing ebpf object and config into package.json...
    $ sudo ecli examples/bpftools/opensnoop/package.json
    TIME     TS      PID     UID     RET     FLAGS   COMM    FNAME
    20:31:50  0      1       0       51      524288  systemd /proc/614/cgroup
    20:31:50  0      33182   0       25      524288  ecli    /etc/localtime
    20:31:53  0      754     0       6       0       irqbalance /proc/interrupts
    20:31:53  0      754     0       6       0       irqbalance /proc/stat
    20:32:03  0      754     0       6       0       irqbalance /proc/interrupts
    20:32:03  0      754     0       6       0       irqbalance /proc/stat
    20:32:03  0      632     0       7       524288  vmtoolsd /etc/mtab
    20:32:03  0      632     0       9       0       vmtoolsd /proc/devices

    $ sudo ecli examples/bpftools/opensnoop/package.json --pid_target 754
    TIME     TS      PID     UID     RET     FLAGS   COMM    FNAME
    20:34:13  0      754     0       6       0       irqbalance /proc/interrupts
    20:34:13  0      754     0       6       0       irqbalance /proc/stat
    20:34:23  0      754     0       6       0       irqbalance /proc/interrupts
    20:34:23  0      754     0       6       0       irqbalance /proc/stat
    ```

    或使用 docker 编译:

    ```shell
    docker run -it -v `pwd`/:/src/ ghcr.io/eunomia-bpf/ecc-`uname -m`:latest
    ```

    编译发布后, 也可以轻松从云端一行命令启动任意 eBPF 程序, 例如:

    ```bash
    wget https://aka.pw/bpf-ecli -O ecli && chmod +x ./ecli     # download the release from https://github.com/eunomia-bpf/eunomia-bpf/releases/latest/download/ecli
    sudo ./ecli https://eunomia-bpf.github.io/eunomia-bpf/sigsnoop/package.json # simply run a pre-compiled ebpf code from a url
    sudo ./ecli sigsnoop:latest # run with a name and download the latest version bpf tool from our repo
    ```

    完整代码在这里: <https://github.com/eunomia-bpf/eunomia-bpf/tree/master/examples/bpftools/opensnoop>

2. 支持根据代码中的注释信息自动生成用户态命令行参数。

    比如需要实现一个 ebpf 程序里面的 pid 过滤器，只需要编写内核态代码，在 eBPF 中声明全局变量，即可自动生成命令行参数：

    ```c
    /// Process ID to trace
    const volatile pid_t pid_target = 0;
    /// Thread ID to trace
    const volatile pid_t tgid_target = 0;
    /// @description User ID to trace
    const volatile uid_t uid_target = 0;
    /// @cmdarg {"default": false, "short": "f", "long": "failed"}
    /// @description target pid to trace
    const volatile bool targ_failed = false;
    ```

    我们会将注释文档的描述信息提取，放在配置文件里面，并且变成 eBPF 应用的命令行参数. 使用方式以跟踪所有 open 系统调用的 opensnoop 为例：

    ```console
    $ sudo ecli  examples/bpftools/opensnoop/package.json -h
    Usage: opensnoop_bpf [--help] [--version] [--verbose] [--pid_target VAR] [--tgid_target VAR] [--uid_target VAR] [--failed]

    Trace open family syscalls.

    Optional arguments:
      -h, --help    shows help message and exits
      -v, --version prints version information and exits
      --verbose     prints libbpf debug information
      --pid_target  Process ID to trace
      --tgid_target Thread ID to trace

    $ sudo ecli examples/bpftools/opensnoop/package.json --pid_target 754
    TIME     TS      PID     UID     RET     FLAGS   COMM    FNAME
    20:34:13  0      754     0       6       0       irqbalance /proc/interrupts
    20:34:13  0      754     0       6       0       irqbalance /proc/stat
    20:34:23  0      754     0       6       0       irqbalance /proc/interrupts
    20:34:23  0      754     0       6       0       irqbalance /proc/stat
    ```

3. 支持自动采集和综合非 ring buffer 和 perf event 的 map，比如 hash map，打印出信息或生成直方图。

    之前使用 ring buffer 和 perf event 的场景会稍微受限，因此需要有一种方法可以自动从 maps 里面采集数据，在源代码里面添加注释即可：

    ```c
    /// @sample {"interval": 1000, "type" : "log2_hist"}
    struct {
        __uint(type, BPF_MAP_TYPE_HASH);
        __uint(max_entries, MAX_ENTRIES);
        __type(key, u32);
        __type(value, struct hist);
    } hists SEC(".maps");
    ```

    就会每隔一秒去采集一次 counters 里面的内容（print_map），以 runqlat 为例：

    ```console
    $ sudo ecli examples/bpftools/runqlat/package.json -h
    Usage: runqlat_bpf [--help] [--version] [--verbose] [--filter_cg] [--targ_per_process] [--targ_per_thread] [--targ_per_pidns] [--targ_ms] [--targ_tgid VAR]

    Summarize run queue (scheduler) latency as a histogram.

    Optional arguments:
      -h, --help            shows help message and exits
      -v, --version         prints version information and exits
      --verbose             prints libbpf debug information
      --filter_cg           set value of bool variable filter_cg
      --targ_per_process    set value of bool variable targ_per_process
      --targ_per_thread     set value of bool variable targ_per_thread
      --targ_per_pidns      set value of bool variable targ_per_pidns
      --targ_ms             set value of bool variable targ_ms
      --targ_tgid           set value of pid_t variable targ_tgid

    Built with eunomia-bpf framework.
    See https://github.com/eunomia-bpf/eunomia-bpf for more information.

    $ sudo ecli examples/bpftools/runqlat/package.json
    key =  4294967295
    comm = rcu_preempt

        (unit)              : count    distribution
            0 -> 1          : 9        |****                                    |
            2 -> 3          : 6        |**                                      |
            4 -> 7          : 12       |*****                                   |
            8 -> 15         : 28       |*************                           |
           16 -> 31         : 40       |*******************                     |
           32 -> 63         : 83       |****************************************|
           64 -> 127        : 57       |***************************             |
          128 -> 255        : 19       |*********                               |
          256 -> 511        : 11       |*****                                   |
          512 -> 1023       : 2        |                                        |
         1024 -> 2047       : 2        |                                        |
         2048 -> 4095       : 0        |                                        |
         4096 -> 8191       : 0        |                                        |
         8192 -> 16383      : 0        |                                        |
        16384 -> 32767      : 1        |                                        |

    $ sudo ecli examples/bpftools/runqlat/package.json --targ_per_process
    key =  3189
    comm = cpptools

        (unit)              : count    distribution
            0 -> 1          : 0        |                                        |
            2 -> 3          : 0        |                                        |
            4 -> 7          : 0        |                                        |
            8 -> 15         : 1        |***                                     |
           16 -> 31         : 2        |*******                                 |
           32 -> 63         : 11       |****************************************|
           64 -> 127        : 8        |*****************************           |
          128 -> 255        : 3        |**********                              |
    ```

    完整代码在这里: <https://github.com/eunomia-bpf/eunomia-bpf/tree/master/examples/bpftools/runqlat>

4. 添加对 uprobe, tc 等多种类型 map 的支持, 允许用标记实现添加额外 attach 信息, 例如:

    ```c

    /// @tchook {"ifindex":1, "attach_point":"BPF_TC_INGRESS"}
    /// @tcopts {"handle":1,  "priority":1}
    SEC("tc")
    int tc_ingress(struct __sk_buff *ctx)
    {
        void *data_end = (void *)(__u64)ctx->data_end;
        void *data = (void *)(__u64)ctx->data;
        struct ethhdr *l2;
        struct iphdr *l3;

        if (ctx->protocol != bpf_htons(ETH_P_IP))
            return TC_ACT_OK;

        l2 = data;
        if ((void *)(l2 + 1) > data_end)
            return TC_ACT_OK;

        l3 = (struct iphdr *)(l2 + 1);
        if ((void *)(l3 + 1) > data_end)
            return TC_ACT_OK;

        bpf_printk("Got IP packet: tot_len: %d, ttl: %d", bpf_ntohs(l3->tot_len), l3->ttl);
        return TC_ACT_OK;
    }
    ```

## 编译方面：编译体验优化、格式改进

1. 完全重构了编译工具链和配置文件格式，回归本质的配置文件 + ebpf 字节码 .o 的形式，不强制打包成 JSON 格式，对分发使用和人类编辑配置文件更友好，同时也可以更好地和 libbpf 相关工具链兼容;
2. 支持 JSON 和 YAML 两种形式的配置文件（xxx.skel.yaml 和 xxx.skel.json），或打包成 package.json 和 package.yaml 进行分发;
3. 尽可能使用 BTF 信息表达符号类型，并且把 BTF 信息隐藏在二进制文件中，让配置文件更可读和可编辑，同时复用 libbpf 提供的 BTF 处理机制，完善对于类型的处理；
4. 支持更多的数据导出类型：enum、struct、bool 等等
5. 编译部分可以不依赖于 docker 运行，可以安装二进制和头文件到 ~/.eunomia（对嵌入式或者国内网络更友好，更方便使用），原本 docker 的使用方式还是可以继续使用；
6. 文件名没有特定限制，不需要一定是 xxx.bpf.h 和 xxx.bpf.c，可以通过 ecc 指定当前目录下需要编译的文件;
7. 把 example 中旧的 xxx.bpf.h 头文件修改为 xxx.h，和 libbpf-tools 和 libbpf-bootstrap 保持一致，确保 0 代码修改即可复用 libbpf 相关代码生态；
8. 大幅度优化编译速度和减少编译依赖，使用 Rust 重构了编译工具链，替换原先的 python 脚本;

在配置文件中, 可以直接修改 progs/attach 控制挂载点，variables/value 控制全局变量，maps/data 控制在加载 ebpf 程序时往 map 里面放什么数据，export_types/members 控制往用户态传输什么数据格式，而不需要重新编译 eBPF 程序。配置文件和 bpf.o 二进制是配套的，应该搭配使用，或者打包成一个 package.json/yaml 分发。打包的时候会进行压缩，一般来说压缩后的配置文件和二进制合起来的大小在数十 kb 。

配置文件举例:

```yaml
bpf_skel:
  data_sections:
  - name: .rodata
    variables:
    - name: min_duration_ns
      type: unsigned long long
      value: 100
  maps:
  - ident: exec_start
    name: exec_start
    data:
      - key: 123
        value: 456
  - ident: rb
    name: rb
  - ident: rodata
    mmaped: true
    name: client_b.rodata
  obj_name: client_bpf
  progs:
  - attach: tp/sched/sched_process_exec
    link: true
    name: handle_exec
export_types:
- members:
  - name: pid
    type: int
  - name: ppid
    type: int
  - name: comm
    type: char[16]
  - name: filename
    type: char[127]
  - name: exit_event
    type: bool
  name: event
  type_id: 613
```

## 下载安装 eunomia-bpf

- Install the `ecli` tool for running eBPF program from the cloud:

    ```console
    $ wget https://aka.pw/bpf-ecli -O ecli && chmod +x ./ecli
    $ ./ecli -h
    Usage: ecli [--help] [--version] [--json] [--no-cache] url-and-args
    ....
    ```

- Install the compiler-toolchain for compiling eBPF kernel code to a `config` file or `Wasm` module:

    ```console
    $ wget https://github.com/eunomia-bpf/eunomia-bpf/releases/latest/download/ecc && chmod +x ./ecc
    $ ./ecc -h
    eunomia-bpf compiler
    Usage: ecc [OPTIONS] <SOURCE_PATH> [EXPORT_EVENT_HEADER]
    ....
    ....
    ```

  or use the docker image for compile:

    ```bash
    docker run -it -v `pwd`/:/src/ ghcr.io/eunomia-bpf/ecc-`uname -m`:latest # compile with docker. `pwd` should contains *.bpf.c files and *.h files.
    ```

## 下一步发展的计划

1. 和更多的社区伙伴合作, 并逐步形成标准化的, 使用配置文件或 Wasm 二进制进行打包分发, 一次编译, 到处运行的 eBPF 程序格式;
2. 和 LMP 社区一起, 完善基于 ORAS, OCI 和 Wasm 的 eBPF 程序分发和运行时标准, 让任意 eBPF 应用均可从云端一行命令拉下来直接运行, 或轻松嵌入其他应用中使用, 无需关注架构, 内核版本等细节;
3. 尝试和 Coolbpf 社区一同完善远程编译, 低版本支持的特性, 以及支持 RPC 的 libbpf 库;
4. 完善用户态 Wasm 和 eBPF 程序之间的互操作性, 探索 WASI 的相关扩展;

## 参考资料

1. [当 Wasm 遇见 eBPF ：使用 WebAssembly 编写、分发、加载运行 eBPF 程序](https://eunomia-bpf.github.io/blog/ebpf-wasm.html)
2. [如何在 Linux 显微镜（LMP）项目中开启 eBPF 之旅？](https://eunomia-bpf.github.io/blog/lmp-eunomia.html)
3. [龙蜥社区 eunomia-bpf 项目主页](https://openanolis.cn/sig/ebpfresearch/doc/640013458629853191)
4. [eunomia-bpf 项目文档](https://eunomia-bpf.github.io/)
5. [LMP 项目](https://github.com/linuxkerneltravel/lmp)

## 我们的微信群
