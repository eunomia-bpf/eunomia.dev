# Why can a BPF uprobe program write a register that decides whether the kernel returns to user mode, and what should the verifier reject?

**Short answer:** the context of a kprobe or uprobe program is the live `struct pt_regs` of the running task, and the verifier's only bound on a write to that context is the size of the structure. Every word of it is writable, including the words the exception-return path reloads into control state: `status` on riscv, `pstate` on arm64, and `cs`, `ss` and `flags` on x86-64. ptrace writes the same structure through per-architecture code that keeps those fields in range. The BPF context check has no such gate.

The practical consequence on riscv is one instruction. Offset 256 in `struct pt_regs` is `status`, and the return path copies it into `sstatus` before `sret`. A single store of `0x100` there sets `SR_SPP`, and the trap then returns to a user address in supervisor mode. It is a real bug, it is not riscv-only, and as of today the fix exists as a diff in a mailing-list reply, not as a merged commit.

## The mechanism

1. **The context check bounds a size, not a field set.** `kprobe_prog_is_valid_access()` accepts any offset while `off >= 0` and `off + size <= sizeof(struct pt_regs)`, rejects misalignment, and on a write sets `prog->aux->kprobe_write_ctx = true`. Nothing else is examined:

   ```c
   if (type == BPF_WRITE)
           prog->aux->kprobe_write_ctx = true;
   ```

   The verifier reaches this function through a table indexed by program type: `__check_ctx_access()` calls `env->ops->is_valid_access()`, and for `BPF_PROG_TYPE_KPROBE` the ops are `kprobe_verifier_ops`. kprobe and uprobe programs share that type, so the same window applies to both. Because there is no `convert_ctx_access`, the verifier also allows a variable offset into the context; the bound is still the structure size.

2. **On riscv the low 32 bytes past the exported register set are control state.** riscv `struct pt_regs` is 288 bytes, and `status` sits at 256, right after `t6`. The exported `user_regs_struct`, which is what BTF and the uapi expose for BPF, is 256 bytes and ends at `t6`. So the writable window is exactly 32 bytes wider than the register set, and `status` is the first word in that extra tail, followed by `badaddr`, `cause` and `orig_a0`.

3. **The return path trusts it.** `ret_from_exception` loads `PT_STATUS(sp)` into `a0`, restores the user stack pointer from it and writes `CSR_STATUS`. `SR_SPP` is `0x00000100` and `user_mode()` is `(regs->status & SR_PP) == 0`, so the same bit both describes where the trap came from and selects where `sret` goes. Setting it makes the kernel return at a user virtual address with supervisor privileges. The reporter's program is one instruction: opcode `0x7a` (store immediate, double word), destination `r1` (the context), offset `0x100`, immediate `0x100`.

4. **arm64 has the same shape at `pstate`.** `struct pt_regs` holds `regs[31]`, `sp`, `pc`, then `pstate` at 264. The kernel exit sequence loads the ELR and SPSR pair from `S_PC`, then does `msr spsr_el1, x22` before `eret`. `pstate` carries the mode and the interrupt masks, so a program that writes it chooses the exception level and the DAIF state of the return. The BPF context here is `user_pt_regs`, which ends with `pstate` itself.

5. **x86-64 has three separate holes.** `cs` is at 136, `flags` at 144, `ss` at 160. The user return path pops the registers, then asserts `testb $3, 8(%rsp)` on the low byte of `CS` and executes `ud2` if the RPL is not 3; a program that writes `cs` therefore turns a return into a fatal assertion instead of a controlled transition. `ss` is copied into the same frame. `flags` is loaded by `iretq`, which takes `IF` and `IOPL` from it: clearing `IF` returns to user with interrupts disabled, and setting `IOPL` to 3 lets the user program issue port I/O.

6. **ptrace already encodes the answer, per architecture.** A debugger writes the same structure and each architecture refuses the fields its return path will reload blindly. x86 `set_segment_reg()` rejects a null selector for `cs` and `ss` and any selector whose RPL is not user, and `set_flags()` keeps unknown bits by masking with `FLAG_MASK`, which is the arithmetic and condition flags plus `NT`, so `IF` and `IOPL` cannot be set. arm64 runs every write through `valid_user_regs()`, which clears the reserved `pstate` bits and forces a valid EL0 state when the value does not form one. riscv simply leaves `status` out of the register set: the regset reads and writes `sizeof(struct user_regs_struct)` bytes, so `PTRACE_GETREGSET` and `PTRACE_SETREGSET` never reach offset 256. Three mechanisms, one rule: a program may set the registers the architecture will reload from the value as given.

7. **`kprobe_write_ctx` is a flag, not a whitelist.** The bit records that a program wrote its context. Exactly one attach path consumes it: `kprobe_multi` attach returns `-EINVAL` with the comment that writing to context is not allowed for kprobes. `uprobe_multi` attach does not test the flag at all, so a multi-uprobe program that writes `pstate`, `cs` or `flags` attaches today. And a single kprobe or uprobe program has no attach-side check either; kprobes documentation has long told probe handlers they may modify `regs->ip` and returns.

8. **The proposed fix moves the rule into the verifier.** The reply that diagnosed the bug sketches `arch_uprobe_regs_writable(off, size)`, a weak function returning false with per-architecture overrides: riscv allows up to `offsetof(struct pt_regs, status)`, arm64 up to `offsetof(struct pt_regs, pstate)`, x86-64 allows the general registers, `ip` and `sp`, and declares 32-bit unsupported. `kprobe_prog_is_valid_access()` would call it inside the `BPF_WRITE` branch and reject before setting the flag. An architecture without an override gets no context writes, which is the conservative default. A maintainer of the uprobe code said the change is his to send.

## Verifying it

On a riscv machine, the whole report reduces to a load, a dump and one store:

```bash
bpftool prog load uprobe_write.bpf.o /sys/fs/bpf/uw autoattach
bpftool prog dump xlated id "$(bpftool prog show name uprobe_write | head -1 | awk -F: '{print $1}')"
```

The dump should show a double-word store to `r1 + 256`. Then check the two ends of the path in the source for the kernel build under test: the context bound is in `kernel/trace/bpf_trace.c`, and the `PT_STATUS` load is in `arch/riscv/kernel/entry.S`. The offsets move between versions, so read them from the build rather than from this page; `asm-offsets` emits `S_PSTATE`, `S_PC` and `S_SP` for arm64, and the x86 offsets follow the order of the structure.

For a non-riscv check, write `pstate` on arm64 or `cs` on x86-64 and look at where the kernel stops: on x86-64 a wrong selector should reach the assertion on the user return path, which needs `CONFIG_DEBUG_ENTRY` only for the kernel-mode check, not for the user-mode one. Keep the program, the translated dump, the kernel build and the exact offset together, because a report without the offset is not reproducible on another version.

No local reproducer was run for this page: the host has no riscv hardware and no `bpftool`, so every claim above is read from source and from the report, and the offsets are arithmetic on the published structure layouts.

## Limits

The fix is a sketch in a reply. No commit was merged in `mainline` or in `bpf-next` at the time of writing, so kernels built today still accept the write, and a page that says otherwise will be wrong within days: check the tree before relying on a rejection.

`arch_uprobe_regs_writable` would make the rule architecture-specific, which is the correct shape but a maintenance load: every architecture that exports pt_regs to BPF has to list its writable registers, and a layout change silently moves the boundary. Architectures without an override lose context writes entirely, which is safe but is a visible behaviour change for programs that legitimately modify a register they own.

The bound is also coarse in one direction: the check is on the byte range of the access, so a store that starts before `pstate` and ends inside it is rejected, while a store of a single word that lands exactly on the field is caught only if the architecture's list stops before it. Sub-word stores and unaligned combinations deserve their own tests, and the reply's diff relies on `off + size <=` comparisons rather than on a table of allowed ranges.

Finally, this is a permission problem inside an already-trusted program class. It does not widen what an unprivileged user can do, and it is not related to `bpf_probe_write_user`; the program is writing the kernel's saved frame, not the process's memory.

## References

- [Report: riscv uprobe writes status](https://lore.kernel.org/bpf/30318.1794850523@30-11-166.wireless.csail.mit.edu/) — the one-instruction program, the offset, and the `sstatus.SPP` consequence.
- [Reply: not only riscv, and the ptrace comparison](https://lore.kernel.org/bpf/DMWUY4M0QXMH.33XAPR76ZXKRT@gmail.com/) — the per-architecture table of exit-path fields and ptrace guards, and the `arch_uprobe_regs_writable()` patch sketch.
- [Reply: the uprobe maintainer takes the change](https://lore.kernel.org/bpf/avt5WnmsPkYkeZrz@krava/) — confirmation that the fix is being sent.
- [Kernel trace program context access](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/trace/bpf_trace.c) — `kprobe_prog_is_valid_access()`, the write flag, and the `kprobe_multi` attach check.
- [Kernel verifier, context accesses](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) — `__check_ctx_access()` and the per-type ops table.
- [Kernel riscv ptrace header](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/riscv/include/asm/ptrace.h) — the structure layout and `user_mode()`.
- [Kernel riscv exception return](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/riscv/kernel/entry.S) — the `PT_STATUS` load and the `CSR_STATUS` write.
- [Kernel riscv status bits](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/riscv/include/asm/csr.h) — `SR_SPP` and `SR_PP`.
- [Kernel riscv ptrace register set](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/riscv/kernel/ptrace.c) — the register set that stops at `t6`.
- [Kernel arm64 exception exit](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/kernel/entry.S) — the ELR/SPSR pair and `msr spsr_el1`.
- [Kernel arm64 ptrace state check](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/kernel/ptrace.c) — `valid_user_regs()` and the forced EL0 state.
- [Kernel x86 user return path](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/entry/entry_64.S) — the `CS` assertion and `iretq`.
- [Kernel x86 ptrace register writes](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/kernel/ptrace.c) — `set_segment_reg()`, `set_flags()` and `FLAG_MASK`.
- [Kernel uprobes core](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/events/uprobes.c) — the resume-time handler that runs the program on the live `pt_regs`.
- [Kernel kprobes documentation](https://docs.kernel.org/trace/kprobes.html) — the long-standing statement that a handler may modify the saved registers.

## Community discussion today

This run read the public bpf mailing list in a visible browser and used the opt-in chat archive as a checked, secondary source. The archive snapshot covered two workspaces and three channels: nothing in the 24-hour window, and two messages in the seven-day window, both non-technical, one a conference meeting-notes announcement and one a contributor introduction. Neither asked a question, so the question above comes from the public list.


The list itself was busy: 42 thread roots dated 2026-11-16 on the first page. Beyond the question above, four threads are worth a reader's time.

**Raw tracepoint register slots under preemption.** Raw tracepoint and tracing programs have no `pt_regs`, so three helpers fill one of three per-CPU slots chosen by a nest counter, assuming callers leave in the reverse order they arrived. Preemption breaks that assumption at task level, and the author's reproducer measures 7 to 18 samples per program that carry the other program's return address out of 1.4 million, plus empty stacks from `bpf_get_stack()` and `bpf_get_stackid()`. The fix disables preemption from get to put. The reviewer's reply accepts the bug and asks a larger question: why keep a per-CPU stack of `pt_regs` at all, when 168 to 384 bytes of stack per call site would remove the shared resource, and he prints a table of `pt_regs` and BPF context sizes for fourteen architecture variants.

**A trusted pointer that is not populated yet.** Two backing-file LSM hooks hand the program a `struct file` whose path is not set: `f_path.dentry` is NULL during allocation and can be cleared again for a file whose open failed. Neither hook is in the untrusted list, so the verifier lets a program pass the path to the trusted-argument kfunc that formats it, and `d_path()` faults on a null dentry on the first open of a file on an overlayfs mount. Adding both hooks to the untrusted list turns the program into a load-time rejection, and the patch is already in the bpf tree as `ab86b62012b9`.

**Verifier limits removed in bulk.** A 21-patch series replaces the verifier's fixed ceilings on map count, subprograms, kfuncs, call frames and instruction count with limits derived from available memory, using the cgroup memory limit as the new bound and resizing the log and jump-fixup structures accordingly. It also relaxes far jumps, makes the dominator computation `O(n log n)`, and adds a large selftest that loads programs beyond the old numbers. The interesting part for users is not the new maxima but the failure mode: a program that used to fail at a named limit should now fail only for lack of memory, which changes both diagnostics and the shape of a denial-of-service budget.

**AF_XDP zero copy through page-pool memory providers.** An eleven-patch series lets a driver install a page-pool memory provider at bind, have the core drive refills through a new `page_pool_refill_done()` instead of NAPI, and deliver provider chunks to a socket without copying, with a per-memory-model capability that says whether the stack may keep a buffer. The numbers on one driver and one queue are 21.2 Mpps receive, 28.9 Mpps transmit and a 1.9 percent cost for `XDP_DROP` at 30 instructions per packet. Still open: how userspace learns the chunk-size constraints before it binds, and that generic XDP and cpumap cannot feed a provider socket.

The remaining threads in the window are small and precise: batch map dumps that replace 200,005 syscalls with five, a cross-endian build fix in the BTF loader for headers longer than the structure, removal of a type tag from per-cpu *variable* declarations where no compiler can record it, and documentation that finally states which program types may mutate a socket map.