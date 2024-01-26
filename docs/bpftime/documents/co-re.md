# Compile Once, Run Everywhere(CO-RE)

You can simply provide the external BTF for userspace host application in your eBPF program:

```c
    LIBBPF_OPTS(bpf_object_open_opts , opts,
    );
    LIBBPF_OPTS(bpf_uprobe_opts, uprobe_opts);
    if (argc != 3 && argc != 2) {
        fprintf(stderr, "Usage: %s <example-name> [<external-btf>]\n", argv[0]);
        return 1;
    }
    if (argc == 3)
        opts.btf_custom_path = argv[2];

    /* Set up libbpf errors and debug info callback */
    libbpf_set_print(libbpf_print_fn);

    /* Cleaner handling of Ctrl-C */
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    /* Load and verify BPF application */
    skel = uprobe_bpf__open_opts(&opts);
    if (!skel) {
        fprintf(stderr, "Failed to open and load BPF skeleton\n");
        return 1;
    }
```

This is the same usage as enabling CO-RE on older kernel versions without BTF information. Also, you need to record the data structs in the BTF info of the eBPF program, for example, if you want `struct data` to be relocated:

```c
#define BPF_NO_GLOBAL_DATA
// #define BPF_NO_PRESERVE_ACCESS_INDEX
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = record)
#endif

struct data {
        int a;
        int c;
        int d;
};

#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute pop
#endif


SEC("uprobe/examples/btf-base:add_test")
int BPF_UPROBE(add_test, struct data *d)
{
    int a = 0, c = 0;
    bpf_probe_read_user(&a, sizeof(a), &d->a);
    bpf_probe_read_user(&c, sizeof(c), &d->c);
    bpf_printk("add_test(&d) %d + %d = %d\n", a, c,  a + c);
    return a + c;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";
```

In fact, the BTF implementation for relocation requires two parts: the compile-time BTF information carried by the BPF program, and the BTF information of the kernel when loading the eBPF program. When actually loading the eBPF program, libbpf will modify potentially incorrect eBPF instructions based on the accurate BTF information of the current kernel, ensuring compatibility across different kernel versions.

Interestingly, libbpf does not differentiate whether these BTF information come from user-space programs or the kernel. Therefore, by merging the user-space BTF information with kernel BTF and provide them to libbpf, the problem is solved.

For more details, please see our blog [Expanding eBPF Compile Once, Run Everywhere(CO-RE) to Userspace Compatibility](../../tutorials/38-btf-uprobe/README.md). The CO-RE can also be used in kernel eBPF for userspace applications.
