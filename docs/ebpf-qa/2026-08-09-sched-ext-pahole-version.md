# Why can an extensible BPF scheduler fail until `pahole` is upgraded?

**Short answer:** because `pahole` is part of the kernel build toolchain, not merely an inspection utility. When `CONFIG_DEBUG_INFO_BTF=y`, the kernel build uses `pahole` to convert DWARF debug information into the BTF embedded in `vmlinux`. A newly compiled kernel can therefore carry incomplete or incorrectly described BTF if it was rebuilt with an encoder version that mishandles a type used by `sched_ext`. Rebuilding with the same inputs reproduces the same metadata; upgrading the encoder and rebuilding changes the kernel BTF that the verifier and libbpf consume.

This does not mean every `sched_ext` load failure is a `pahole` bug. The same visible symptom can come from a missing kernel option, a scheduler built against a different `sched_ext` revision, an unsupported kfunc, or an ordinary verifier rejection. The useful diagnostic is to treat the running kernel, its BTF, and the BPF object as one versioned set.

## Why the encoder version can change the result

BTF describes kernel types and function signatures in a compact form. libbpf uses the running kernel's BTF for CO-RE relocation, and the verifier uses BTF when checking typed kfunc and `struct_ops` interfaces. `sched_ext` depends heavily on both: a scheduler supplies a `struct sched_ext_ops` implementation and calls scheduler kfuncs whose types must match the running kernel.

The kernel documentation describes `pahole` as the DWARF-to-BTF converter. Its 1.31 release fixed several BTF-encoder boundaries, including function selection for stack-passed structures with unusual alignment, inference of alignment for zero-length arrays and bitfield-adjacent arrays, and BTF deduplication through an updated libbpf. Those are metadata-generation fixes. They do not change the scheduler source, but they can change which functions and layouts are safely represented in the kernel's BTF.

The important boundary is therefore:

1. the compiler emits DWARF while building the kernel;
2. `pahole` converts that DWARF into `.BTF`;
3. the built kernel exposes its BTF at `/sys/kernel/btf/vmlinux`;
4. libbpf relocates the scheduler object against that runtime BTF; and
5. the verifier checks the resulting `struct_ops` programs and kfunc calls.

An older encoder can make step 2 the failure's root cause even when the rejection appears only at step 5.

## A diagnostic sequence that separates the causes

First, record the exact build and runtime inputs before changing anything:

```console
$ uname -r
$ pahole --version
$ grep -E 'CONFIG_(DEBUG_INFO_BTF|SCHED_CLASS_EXT)=' /boot/config-$(uname -r)
$ test -r /sys/kernel/btf/vmlinux && echo runtime-BTF-present
```

Confirm that the `pahole --version` result is the binary used by the kernel build, not just a newer binary installed afterward. Build logs and the kernel build environment are better evidence than the current shell's `PATH`.

Next, inspect the produced and running artifacts:

```console
$ readelf -S ./vmlinux | grep -E '[.]BTF([.]ext)?'
$ bpftool btf dump file /sys/kernel/btf/vmlinux format raw > /tmp/runtime.btf.txt
$ bpftool btf dump file ./vmlinux format raw > /tmp/built.btf.txt
```

The two dumps should describe the same kernel image. If the scheduler is meant for the running kernel, regenerate `vmlinux.h` from that runtime BTF and rebuild the BPF object. Then load a minimal scheduler from the same `scx` and kernel revision before testing a larger policy. This distinguishes a broken base interface from a policy-specific verifier failure.

Finally, preserve the complete verifier log and classify the rejection:

- a missing `CONFIG_SCHED_CLASS_EXT` or missing runtime BTF is a kernel configuration problem;
- a missing kfunc or changed `struct sched_ext_ops` member is usually a source/API revision mismatch;
- a CO-RE relocation failure points to differences between the object and runtime BTF;
- an invalid argument, state, or control-flow rejection can be an ordinary scheduler bug; and
- a failure that disappears only when the same kernel source is rebuilt with a newer `pahole` strongly implicates BTF generation, but the BTF diff is still needed to identify the exact encoder defect.

Do not copy another machine's `vmlinux` or BTF into place as a shortcut. The correction is to rebuild the kernel with the intended toolchain, boot that exact image, and verify that `/sys/kernel/btf/vmlinux` belongs to it.

## Operational limit

The `sched_ext` documentation explicitly states that its BPF-facing ABI has no stability guarantee across kernel versions. A corrected encoder removes one source of bad metadata; it does not make a scheduler portable across arbitrary kernel and `scx` revisions. Distribution-kernel testing remains important because compiler, configuration, backports, BTF generation, and the scheduler interface all vary together. The upstream `scx` work to run verifier tests against several distribution kernels is a practical model for catching those combinations before deployment.

## References

- [`pahole` v1.31 release notes](https://github.com/acmel/dwarves/releases/tag/v1.31)
- [Linux kernel documentation: BPF Type Format](https://docs.kernel.org/bpf/btf.html)
- [Linux kernel documentation: `bpftool-btf`](https://docs.kernel.org/bpf/bpftool-btf.html)
- [Linux kernel documentation: `sched_ext`](https://docs.kernel.org/scheduler/sched-ext.html)
- [`scx` pull request: verifier tests against distribution kernels](https://github.com/sched-ext/scx/pull/3700)
- [BPF mailing list: aggregate return values up to 16 bytes](https://lore.kernel.org/bpf/20260808190322.1896580-1-yonghong.song@linux.dev/T/#t)
- [BPF mailing list: arena fault-in under `memory.max`](https://lore.kernel.org/bpf/20260808140720.293604-1-jiayuan.chen@linux.dev/T/#t)
- [BPF mailing list: arena arguments for kfuncs and `struct_ops`](https://lore.kernel.org/bpf/178618382564.2611241.3179867674703496337.git-patchwork-notify@kernel.org/T/#t)
- [BPF mailing list: `tracing_multi` link support](https://lore.kernel.org/bpf/20260809150111.45000-1-leon.hwang@linux.dev/T/#t)

## Community discussion today

Today's visible review covered 6 approved communities and 15 allowlisted channels or public pages. All were accessible. Identity, channel names, message links, exact times, and deployment-specific details have been removed. Several project channels contained only automated development notices, and several discussion channels had no substantive technical activity in the 24-hour window; those quiet channels are included in the coverage count rather than presented as engagement.

### Kernel BTF is a build artifact, not a static prerequisite

The clearest practitioner thread was a scheduler that still failed after trying multiple freshly built kernels but worked after upgrading the BTF encoder and rebuilding again. The mechanism is the build chain described above: changing kernel source or configuration does not repair metadata if the same encoder continues to emit it. The immediate resolution path is to record the encoder used during the build, compare the built and runtime BTF, regenerate the BPF-side type header, and reduce the load attempt to a minimal scheduler with the full verifier log. What remains unresolved without an artifact-level diff is which exact type triggered the failure; the 1.31 release notes establish several relevant alignment and deduplication fixes, but they are not proof that one specific fix caused this instance.

The wider operational lesson is that a kernel compatibility matrix must include the toolchain. The upstream distribution-kernel verifier work tests complete combinations rather than assuming that a kernel version alone identifies behavior. That is especially important for `sched_ext`, whose BPF-facing interface can change between kernels.

### BPF type expressiveness is still moving

The public kernel-development archive had active work on returning aggregates of up to 16 bytes from kfuncs, using a register pair and extending verifier precision, liveness, JIT, BTF-reliability, and selftest handling together. The practical symptom for developers is that a C signature that looks ordinary may not yet have a valid BPF calling convention on every architecture or program context. The diagnostic path is to check the documented kernel version and architecture support, then inspect both the BTF signature and verifier log instead of treating the C prototype as the complete contract. The open boundary is callback support: the series deliberately rejects callbacks returning more than 8 bytes while enabling the broader kfunc case.

Related arena work covered typed arena arguments for kfuncs and `struct_ops`, plus memory-limit behavior while faulting arena pages. These discussions reinforce the same rule as the `pahole` incident: BTF must express the address-space and type contract before the verifier and JIT can enforce it. For an arena failure, separate type/relocation rejection from runtime allocation pressure, inspect the cgroup memory limit and fault path, and reduce the program to the relevant typed argument or allocation. The public threads leave architecture coverage and final merge state as moving boundaries rather than promises for an already deployed kernel.

### Multi-attach tracing needs transactional failure handling

Another active upstream series proposed a tracing link that attaches BPF programs across multiple targets, with explicit selftests for cookies, link information, tail calls, failure cases, and rollback. The concrete operational risk is partial attachment: without transactional behavior, a failed multi-target operation could leave an observer covering only part of the intended surface. A useful validation therefore enumerates the resulting link information, deliberately injects one invalid target, and confirms that rollback leaves no residual attachments. Until the interface lands in the target kernel and libbpf version, applications should keep their existing per-target attachment and cleanup logic rather than assuming the proposed multi-link semantics.

The public forum had no new post inside the daily window; its latest practitioner material concerned profiling and verifier examples from earlier in the week. The two observability channels and the general eBPF chat were also quiet inside the window. They were reviewed and counted as quiet, not used as evidence for today's question.
