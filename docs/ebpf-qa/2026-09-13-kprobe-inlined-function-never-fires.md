# Why does a kprobe on a function never fire when the compiler inlined it?

**Short answer:** a symbol-based kernel probe attaches only to a function that
still exists as a standalone symbol at attach time. When the compiler inlines a
static or small function — or the code forces `__always_inline` — that symbol
disappears from kallsyms and no longer carries an ftrace `fentry`/`mcount` call
site, so the attach step fails at registration instead of silently sitting idle.
For a kprobe event the write to `kprobe_events` is rejected with the diagnostic
`Invalid probed address or symbol`; a BPF `fentry`/`fexit` or `kprobe.multi`
attach has no ftrace call site to bind. The fix is not to retry the same probe:
force an out-of-line copy of the target (`noinline`, GCC `-fno-inline`,
`-fkeep-inline-functions`), probe a caller or wrapper the compiler did keep, or
probe by a concrete address. This is a different failure from the kprobe
blacklist, where the symbol exists but kprobes refuses to break it.

## Two registries decide attachability

A kernel function is probeable only if it appears in a registry that the attach
path consults. Inlining knocks the function out of both at once.

The first registry is **kallsyms**. A kprobe event resolves its symbol through
`kallsyms_lookup_name()` in the kernel tracing code; if the lookup yields no
address, symbol validation returns `-ENOENT` before registration is even
attempted. The write to the `kprobe_events` file then fails, and the trace-probe
log names the reason as `Invalid probed address or symbol`. The kernel needs
`CONFIG_KALLSYMS` (and often `CONFIG_KALLSYMS_ALL`) for this resolution to work
at all.

The second registry is the **ftrace instrumented-function set**. An ftrace site
is the per-function call site — "the instruction pointer of the function that is
being traced (where the fentry or mcount is within the function)". The tracefs
file `available_filter_functions` "lists the functions that ftrace has processed
and can trace". A `fprobe` (which backs BPF `kprobe.multi`) is "a function
entry/exit probe based on the function-graph tracing feature in ftrace", and a
BPF `fentry`/`fexit` trampoline resolves its target with `ftrace_location()`; if
that returns nothing, the trampoline has nothing to bind. A function the
compiler inlined away has no such call site in its callers.

BTF does not create an attach point. `BTF_KIND_FUNC` "represents a defined
subprogram" — it describes and types defined subroutines, it does not turn
inlined code back into a standalone, instrumentable function instance. So all
three artifacts a probe could target — the kallsyms symbol, the ftrace call
site, and the BTF function entry — vanish together with the out-of-line copy.

One branch is not a failure: a **module-qualified** symbol whose module is not
loaded yet is deferred, not rejected. The kernel warns "This probe might be able
to register after target module is loaded" and keeps the event so it can be
retried when the module appears. That is distinct from a core symbol that simply
does not exist.

## How to verify which case you are in

```sh
# 1. Does the symbol exist at all? (kprobe-able through kallsyms)
grep -w my_func /proc/kallsyms

# 2. Is it in the ftrace instrumented set? (fentry/fprobe-able)
grep -w my_func /sys/kernel/tracing/available_filter_functions
grep -w my_func /sys/kernel/tracing/available_filter_functions_addrs

# 3. kprobe event attach test: a missing symbol rejects the write
echo 'p:my_probe my_func' > /sys/kernel/tracing/kprobe_events
dmesg | grep -E 'Invalid probed address|module is loaded'

# 4. Is there a BTF entry? An inlined-only function has no FUNC entry
bpftool btf dump file /sys/kernel/btf/vmlinux | grep -w my_func

# 5. Prove the inlining: no standalone body in the object
objdump -d vmlinux | grep -A4 my_func

# 6. perf probe enumerates inlined DWARF instances by default (needs debuginfo)
perf probe -v --add='my_func' -k /path/to/vmlinux
```

If step 1 and step 2 both come up empty while step 5 shows the body only inside
another function's disassembly, the function was inlined and there is no
standalone probe target.

## Forcing a probeable copy

The remedy is compiler-side, not a different probe type:

- Mark the function `noinline`. The clang attribute reference states the
  attribute "suppresses the inlining of a function at the call sites"; GCC
  documents the same per-function exemption.
- Build with GCC `-fno-inline`, which does "not expand any functions inline
  apart from those marked with the `always_inline` attribute".
- Keep an out-of-line copy with GCC `-fkeep-inline-functions`, which emits
  static functions declared `inline` into the object "even if the function has
  been inlined into all of its callers" (clang has no `-fno-inline`; its form is
  `-fno-inline-functions` / `-fkeep-inline-functions`).
- Or change the target: probe a non-inlined caller or wrapper, probe by the
  address where the inlined code actually landed inside its caller, or let
  `perf probe` resolve an inlined instance from DWARF, where `-N`/`--no-inlines`
  limits the search to non-inlined functions.

## The limitation that decides it

You cannot attach a symbol-based probe to a function that the compiler did not
emit as a standalone symbol, and you cannot detect this at probe-creation time
by expecting the probe to "just never fire" — the actual behavior depends on the
path. A kprobe event write is rejected with `Invalid probed address or symbol`;
a BPF `fentry`/`fexit`/`kprobe.multi` attach fails because there is no ftrace
call site; a deferred module-qualified symbol instead succeeds and waits. Do not
confuse this with the kprobe blacklist, where the symbol exists but kprobes
refuses to break it (functions annotated `__kprobes`/`nokprobe_inline` or marked
`NOKPROBE_SYMBOL`): inlining means the symbol is absent, the blacklist means it
is present but forbidden. Verify symbol existence in kallsyms and in
`available_filter_functions` first; only then decide whether to recompile with
`noinline` or move the probe to a caller.

## References

- [The Linux kernel documentation: Kernel Probes (Kprobes)](https://docs.kernel.org/trace/kprobes.html)
- [The Linux kernel documentation: Kprobe-based Event Tracing](https://docs.kernel.org/trace/kprobetrace.html)
- [The Linux kernel documentation: ftrace — Function Tracer](https://docs.kernel.org/trace/ftrace.html)
- [The Linux kernel documentation: Using ftrace to hook to functions](https://docs.kernel.org/trace/ftrace-uses.html)
- [The Linux kernel documentation: Fprobe-based Event Tracing](https://docs.kernel.org/trace/fprobetrace.html)
- [The Linux kernel documentation: BPF Type Format (BTF)](https://docs.kernel.org/bpf/btf.html)
- [Linux kernel source: `kernel/trace/trace_kprobe.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/trace/trace_kprobe.c)
- [Linux kernel source: `kernel/trace/trace_probe.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/trace/trace_probe.h)
- [Linux kernel source: `kernel/bpf/trampoline.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/trampoline.c)
- [perf-probe(1)](https://man7.org/linux/man-pages/man1/perf-probe.1.html)
- [GCC: Options That Control Optimization](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
- [Clang: Attributes in Clang](https://clang.llvm.org/docs/AttributeReference.html)

## Community discussion today

The monitored window was not technical. Over the previous 24 hours the two
opt-in Slack archives returned a small fallback set consisting only of meeting
logistics — a recurring scheduling note, a co-presenting thank-you, and a
session link — with no eBPF question, symptom, or design dispute. The two
allowlisted chat workspaces had no visible browser session this run, and the
public mailing-list and forum archives were not reviewed. Those sources are
recorded as unavailable coverage, not quiet. Because the readable archive
yielded no question, this Q&A is grounded in public primary documentation rather
than a community message: the question is a recurring practitioner problem —
a dynamic probe that attaches to the wrong target, or appears to attach but
never fires, because the compiler inlined the function away — verified against
the kernel tracing documentation, the upstream kprobe and trampoline source, and
the `perf probe` and compiler option references. No private text, identity,
channel, or link is reproduced here.
