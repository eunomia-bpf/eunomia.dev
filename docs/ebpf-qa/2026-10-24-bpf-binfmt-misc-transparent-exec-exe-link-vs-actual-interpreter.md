# When a BPF binfmt-misc handler runs an exec in transparent mode, why do the process exe link, comm, and command line still describe the binary instead of the interpreter that actually runs?

Because the kernel pins "identity" to the binary on purpose. A BPF binfmt-misc handler picks the interpreter through BPF-selected fields on `struct linux_binprm`; when the selection carries the transparent flag, the exec path swaps the *code that runs* to that interpreter but leaves the process identity — the `exe` link, the thread `comm`, and the argument vector — describing the binary. The interpreter decision is stashed in `bprm->bpf_interp`, `bprm->bpf_interp_file`, and `bprm->bpf_flags`, translated into `MISC_FMT_*` bits and then zeroed out inside the exec path. After exec completes, no user-visible interface and no tracepoint records which BPF handler made the choice or with which flags.

That is the whole reason the observation gap is structural, not an oversight in one place: the selection is a load-time, in-kernel fact that is deliberately not left on the process, so the standard exec observability tools are blind to the BPF side of it by design.

## The mechanism

1. The BPF handler is a `struct binfmt_misc_ops` struct_ops map activated by a `'B'` binfmt-misc entry. Its load program is the only place the selection kfuncs `bpf_binprm_set_interp`, `bpf_binprm_select_interp`, `bpf_binprm_set_interp_arg`, and `bpf_binprm_set_flags` are callable — the verifier restricts them to the map's load program. The chosen interpreter is stashed in `bprm->bpf_interp` / `bprm->bpf_interp_file` and the flags in `bprm->bpf_flags`.

2. `load_misc_binary` calls `entry_invocation_flags`, which translates the BPF flags into the `MISC_FMT_*` invocation bits — the transparent flag maps to `MISC_FMT_TRANSPARENT` together with `MISC_FMT_OPEN_BINARY` — and then zeroes `bprm->bpf_flags` so the flags do not accumulate into a nested interpreter level. On the transparent path it skips `build_interp_argv`, calls `bprm_change_interp` to set `bprm->interp` to the interpreter string, and sets `bprm->interpreter` to the interpreter file with `BINPRM_FLAGS_TRANSPARENT_INTERP`.

3. The `exec_binprm` loop then runs the next round with `bprm->file = bprm->interpreter` — the interpreter becomes the next round's file. When `bprm->have_execfd` is set, the original binary is stashed as `bprm->executable` for the `AT_EXECFD` hand-over to the interpreter, and the loop caps interpreter depth at 5 before returning `-ELOOP`.

4. `bprm_identity_file` returns `bprm->executable` — the binary — whenever `BINPRM_FLAGS_TRANSPARENT_INTERP` is set, and `set_mm_exe_file` plus the comm derivation both feed from that. So `/proc/PID/exe` and `comm` present the binary while `argv` is left untouched and the running code is the interpreter reached through `AT_EXECFD`. The transparent mode is, in the kernel's own comment, the path that "update[s] interp for the next round" so the exec "look[s] like a direct execution."

## How to verify and debug

1. Read the exec identity directly: compare `/proc/PID/exe` plus `comm` against `/proc/PID/maps` and the `AT_EXECFD` auxv entry. In transparent mode the exe link, `cmdline`, and `comm` all describe the binary while the running code is the interpreter.

2. Compare the two sched exec events on the same exec: `sched_prepare_exec` carries the `interp` path but still not the handler name or the flags, while `sched_process_exec` — the event most tracers key off — carries only `filename` plus `pid` plus `old_pid`. The two report different things about the same exec, and the BPF handler name and selection flags are absent from both.

3. Reproduce with a BPF handler: register the `'B'` handler, create a `D`-state entry, bind an interpreter, then run an exec through it. The binary's `exe` link stays, the argv stays, and the running code is the interpreter.

4. Decode the observed flag value: a selection recorded as `0x48000000` is `MISC_FMT_TRANSPARENT` or `MISC_FMT_OPEN_BINARY` — the translation in `entry_invocation_flags` adds the `OPEN_BINARY` implication that the transparent flag carries.

## The limitation

- The blind spot is not total: `sched_prepare_exec` does report the interpreter path. What no existing event or interface exposes is the BPF handler name and the selection flags.

- The identity behavior is intentional, not a bug. Transparent mode exists so the exec presents as the binary, which is why the exe link and comm follow the binary and `argv` is preserved.

- The RFC's proposed default-off `binfmt_misc_select` tracepoint is observability, not enforcement. It reports the selection after the fact, and tracefs access is root-only, so root can turn it off.

- A BPF binfmt-misc handler is registered in its own user namespace and is not inherited from ancestor user namespaces, so which handler applied depends on the namespace the exec ran in.

- Loader mode is a different mechanism: it substitutes the loader for the binary's `PT_INTERP` and runs the binary natively, excluding every other flag — it is not the transparent path.

## References

- [Public bpf mailing-list RFC: a default-off `binfmt_misc_select` tracepoint](https://lore.kernel.org/bpf/21bBRWi4dDGR7l5TdmShL09exFRFwQA8c-1U_QlNoY2L7LGzDCNfypM9VECtuwD7mkrG1Cw8uXSLLdHmXsYsMARBbGn-mIl3Mutmeku-2zg=@proton.me/) — a public bpf-list RFC proposing a default-off tracepoint after the BPF handler's interpreter selection, reporting filename, interpreter, handler, masked flags, and pid, explicitly as observability rather than enforcement.
- [Kernel `fs/binfmt_misc_bpf.c`](https://github.com/torvalds/linux/blob/master/fs/binfmt_misc_bpf.c) — the BPF struct_ops handler kfuncs that stash the interpreter and flags.
- [Kernel `fs/binfmt_misc.c`](https://github.com/torvalds/linux/blob/master/fs/binfmt_misc.c) — `load_misc_binary`, `entry_invocation_flags`, and the `MISC_FMT_*` flag bits.
- [Kernel `fs/exec.c`](https://github.com/torvalds/linux/blob/master/fs/exec.c) — `bprm_identity_file`, `bprm_change_interp`, and the `exec_binprm` loop.
- [Kernel `include/trace/events/sched.h`](https://github.com/torvalds/linux/blob/master/include/trace/events/sched.h) — `sched_process_exec` versus `sched_prepare_exec` field lists.

## Community discussion today

A public bpf mailing-list RFC proposes a default-off `binfmt_misc_select` tracepoint that fires after the BPF handler has selected its interpreter, reporting the filename, interpreter, handler, masked flags, and pid. It is framed as observability for the BPF-side selection — the exact fact that, as things stand, no existing tracepoint or user-visible interface exposes. The RFC is public and open for discussion on the list.

This run's archive-backed sources produced 0 messages in the 7-day window (the two opt-in archive sources were empty at snapshot time), so the question above was sourced from the permitted public bpf mailing list, reviewed through its public atom feed, and the RFC thread was used as the primary reference. The Discord channels for eunomia-bpf and sched-ext and r/eBPF were not covered this run (no visible-browser session was available), so they are marked not-covered rather than quiet.
