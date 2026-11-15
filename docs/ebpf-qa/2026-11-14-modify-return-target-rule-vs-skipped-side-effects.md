# Why can a BPF modify-return program fail a kernel function the kernel never listed, and what decides whether skipping it is safe?

Because the attach rule is moving from a hand-maintained allowlist to a property the build can
check. Today a modify-return program attaches to a kernel function only if that function is in the
error-injection list, which is filled from explicit `ALLOW_ERROR_INJECTION` markers, or if its name
starts with `security_`; anything else is rejected with `X() is not modifiable`. A series applied
to the BPF tree this week replaces that with a build-time scan: with the new option on,
`resolve_btfids` reads the linked machine code, finds every function whose return register can hold
an error code, and records those codes as a declaration tag on the function. A program may then
attach to any tagged function, and the verifier requires its return value to be `0` (call the
function as usual) or one of the tagged codes (skip the call and hand the caller that error). What
decides whether skipping is safe is not the scan but a deliberately narrow rule about what the
function has already done before it returns the error: if every path to the error returns through a
call to a void function on a pointer the function received, that error is not offered as a target,
because a skip leaves the work undone. Measured on the author's own injection harness, one error
injected into one tagged function damages the kernel in about one function out of 180.

## The mechanism

1. **The current gate is a list, not a proof.** The verifier's attach path for modify-return
   programs accepts the target if the function's address is inside the error-injection table, or if
   the name begins with the security-hook prefix. The table is filled at init from a linker section
   of markers: the syscall wrapper macros put one on every syscall symbol, a few subsystems mark
   specific functions of their own, and modules contribute theirs on load. That is why the declared
   targets number about 957 syscall entries and 312 security hooks with only about 21 others, and why
   the table says nothing about whether a given function is a reasonable place to fail.

2. **A skip is not a failure of the function.** The trampoline makes the distinction concrete. It
   runs each modify-return program, stores the program's return value in a stack slot, and compares
   that slot against zero; a non-zero value branches straight past the call to the original function,
   straight to the fexit programs. The caller then sees the program's value as the function's return
   value. So a program that returns an error does not make the function fail, it makes the function
   never run, and the caller's cleanup path runs against an object the callee never touched.

3. **What the build finds.** The scan runs where BTF is already being post-processed: after `pahole`
   has produced BTF from the compiler's debug information, `resolve_btfids` is handed the same ELF
   and decodes each function's machine code with LLVM's MC layer. Backward data flow collects the
   constants in the range `-4095..-1` that can reach the return-value register; a forward may-analysis
   asks whether a path to that return avoids the exception's call. The result is a declaration tag
   whose value is the sorted list of those error codes.

4. **Which constants count as errors.** Constants of `-512` and below, plus `-EAGAIN`, `-EMSGSIZE`
   and `-EINPROGRESS`, are dropped: the first group never reaches user space, and the second group
   tells the caller to retry, continue or wait rather than that the call failed. A function whose
   only such constant is `-1` gets no tag, because the machine code cannot tell the `-EPERM` of a
   permission check from the `-1` of "not found" or of a comparator. Return width decides the rest:
   `mov $0xffffffea,%eax` is `-22` through a 4-byte return, and `4294967274` through an 8-byte one,
   so an error of a function that returns `long` must be sign-extended in the full register.

5. **Which functions are skipped before scanning.** A function that no program could attach to is
   never tagged: BTF must say it returns an integer or enum of 4 or 8 bytes, exactly one ELF symbol
   must carry its name, its parameters must be at most twelve of at most sixteen bytes with no
   variadic argument, it must live outside init text, and its entry must carry the nop or the call
   to `__fentry__` that ftrace patches, which excludes `notrace` functions. On the author's builds
   the series tags 12096 functions for x86-64 compiled with clang, 11432 with GCC, and 10954 on
   arm64; one intermediate build reports 12076 tags out of 78350 functions, with none of the 35445
   functions it scanned failing to decode.

6. **The exception is about side effects, not about errors.** The rule says: an error is not allowed
   if every path to its return calls a void function on a pointer that is an argument of the function
   or the address of one of that argument's members, not a pointer loaded through it. In source
   terms that is a lock released, a reference dropped, an object freed or initialized, and then
   `return -EBUSY`. A caller that gets `-EBUSY` from the function itself knows the work was done; a
   caller that gets `-EBUSY` from a skipped function finds the lock still held. The tool sees this in
   the machine code, so the source needs no `__releases()` annotation and no name-based list.
   Concretely, in the series' own example the compiler emits a load of the argument, a `lea` of the
   lock field, a call to a function that BTF says returns void, and a move of `-EBUSY` into the
   return register; the `-EBUSY` is removed, and since it was the only error the function ends up
   with no tag at all.

7. **What the verifier then checks.** Attaching to a tagged function installs a per-program list. At
   every exit the program's return register must be a scalar whose whole range lies inside `{0}`
   plus that list, and the check is over the range, not over the observed value, so a path that
   could produce an unlisted code fails to load. Tail calls are refused outright, because a program
   array puts no constraint on what the successor returns. For a target that returns 8 bytes, the
   program also requires the JIT, because the interpreter propagates only the low 32 bits and a
   value such as `-14` would reach the caller as `4294967282`: not zero, so the call is skipped, and
   not an error either. For a target that returns 4 bytes the 32-bit range decides only when the
   high half of the register is known to be zero, matching what the caller of such a function reads.

8. **Where the tag is looked up.** Only the attaching BTF object's own types are searched. A module
   can name a function that also exists in the kernel's BTF, and the kernel's tag says nothing
   about the module's copy, which is different machine code. Because the build sorts BTF by name,
   the tags sit together and a binary search finds the first one.

## Verifying it

1. **Ask the running kernel what it tagged.** With the option built in, the tag is a declaration
   tag on the function's `FUNC` entry, so the raw dump shows it. A function that returns errors and
   has no tag is the interesting case: check steps 4 to 6 before calling it a bug.

   ```console
   $ bpftool btf dump file /sys/kernel/btf/vmlinux format raw \
       | grep "DECL_TAG 'bpf_fmodret_retval:"
   [12345] DECL_TAG 'bpf_fmodret_retval:-95,-29,...' type_id=6789 component_idx=-1
   ```

   The tag's `type_id` is the `FUNC` entry it describes; dumping that identifier names the function.

2. **Ask which functions an unpatched kernel accepts.** The old gate is observable without a
   rebuild: with function-level error injection enabled, the table is exported through debugfs, one
   line per marker with the address and the error type the marker declared.

   ```console
   $ cat /sys/kernel/debug/error_injection/list
   ```

3. **Read the rejection instead of guessing.** A program whose target is in neither the table nor
   the security namespace fails to load with the function name in the message; a program whose
   return range leaves the list fails with the list printed in the log. The two errors tell you
   which half of the contract you broke.

4. **Expect BTF type identifiers to move.** The tags are additional types in vmlinux BTF, so every
   identifier above the first tag shifts. The series carries its own example: one helper's
   identifier goes from 95497 to 106933, which broke a selftest that matched a fixed-width log
   column. Anything that hard-codes a BTF identifier across the option being enabled is relying on
   an accident, not on an interface.

5. **Check the trampoline's contract before blaming the program.** A skip is decided by comparing
   the program's return value against zero in the generated trampoline, before the original call. If
   a program appears to run while the function's side effects are missing, that comparison is the
   place to look, and the same stack slot is what fexit programs read afterwards.

## Limits

- **This is in the BPF tree, not in a release.** The series was applied to the BPF next tree on the
  day of this article; the option has no default, is absent from `defconfig`, and mainline does not
  have it yet. Read the tag format and the numbers as the current shape of the proposal.

- **The damage rate is one measurement, not a guarantee.** One error injected once into each of 4401
  net, fs and mm functions, on a workload of 530 tests and scripts, most of them selftests: 50 cases
  ended in a panic, an oops, a `BUG()`, a hang or a lockdep report, and the exception removes 27 of
  those. Across all 8027 injections, 48 percent give user space an error and 50 percent give
  nothing. The author is explicit that the rule does not aim at full safety.

- **The exception is narrow by design, and about 114 damaging functions remain.** Across four runs,
  169 functions damaged the kernel and the exception removed 55 of them; the rest are left for a
  name-based deny list that is not part of the series. A function whose cleanup goes through a
  value-returning helper keeps its tag, because that call could be the one that failed.

- **A tag does not mean the failure is harmless.** A function can return an error legitimately and
  still be a bad place to skip for a given caller: failing a call that the caller expected to
  succeed is a loss of function the tag cannot see. That is the stated purpose as well as the stated
  risk.

- **Cost and coverage.** The build host needs a C++17 compiler and the headers and libraries of LLVM
  17 or newer whatever compiler built the kernel, the option covers x86-64 and arm64 only, vmlinux
  BTF grows by about 2.8 percent, and the tool's time on vmlinux goes from 0.31 s to 0.9 s after
  4.3 s of `pahole`.

## References

- [Series cover letter, modify-return targets for functions that return errors](https://lore.kernel.org/bpf/20261114051805.683966-1-alexei.starovoitov@gmail.com/) — the target rule and its exception, the damage table per subsystem, and the cost of the option.
- [Patch, tagging the functions that return errors](https://lore.kernel.org/bpf/20261114051805.683966-6-alexei.starovoitov@gmail.com/) — the tag format, the constants that count as errors, and the attachability filters.
- [Patch, allowing modify-return on tagged functions and checking the return value](https://lore.kernel.org/bpf/20261114051805.683966-10-alexei.starovoitov@gmail.com/) — the exit-value check, the tail-call refusal, and the 32-bit range rule.
- [Patch, requiring the JIT for tagged 64-bit targets](https://lore.kernel.org/bpf/20261114051805.683966-11-alexei.starovoitov@gmail.com/) — the interpreter's 32-bit return value and the resulting skip.
- [Patch, selftests for the tagging rules](https://lore.kernel.org/bpf/20261114051805.683966-12-alexei.starovoitov@gmail.com/) — the function shapes each rule accepts or rejects, and the run subtest that observes a skip.
- [Applied-to-bpf-next notice for the series](https://lore.kernel.org/bpf/179468100783.2330384.13747727141754933715.git-patchwork-notify@kernel.org/) — the tree and the commit list for the applied state.
- [Kernel `kernel/bpf/verifier.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) — the current attach gate, the error-injection table lookup, and the security-prefix fallback.
- [Kernel `lib/error-inject.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/lib/error-inject.c) — how the error-injection table is filled from markers and modules, and its debugfs listing.
- [Kernel `include/asm-generic/error-injection.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/asm-generic/error-injection.h) — the marker macro and the four error types it can declare.
- [Kernel `include/linux/syscalls.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/syscalls.h) — the per-syscall markers that make syscall symbols attachable today.
- [Kernel `arch/x86/include/asm/syscall_wrapper.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/include/asm/syscall_wrapper.h) — the architecture wrapper that marks each syscall entry point.
- [Kernel `arch/x86/net/bpf_jit_comp.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/net/bpf_jit_comp.c) — the generated trampoline that stores the program's return value and branches past the original call on a non-zero value.
- [Kernel `Documentation/fault-injection/fault-injection.rst`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/fault-injection/fault-injection.rst) — the two requirements an error-injectable function must satisfy, including the no-state-change rule.
- [Kernel `kernel/bpf/Kconfig` in the bpf-next tree](https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git/plain/kernel/bpf/Kconfig) — the new option's dependencies, architecture coverage and host tool requirements.

## Community discussion today

This run reviewed the two chat workspaces that opted in to the private read-only archive, three
allowlisted channels in total, and read the public bpf mailing list and a practitioner forum in a
visible browser as the public surfaces. The archive was empty in the strict sense: no message in the
24-hour window, and its seven-day window held ten messages in one workspace and none in the other,
all of them logistics for a conference maintainer meeting and for project talks at co-located events,
one noting a third party presenting network traces combined with continuous profiles for threat
detection. None was a technical question, so the answer above comes from the mailing list, which
carried 31 new thread roots in the day's window. The four allowlisted chat workspaces could not be
read at all: the two Discord invites open a join form with a member count and no channel list, and
the two Slack invites open an invitation-acceptance page, so all four are recorded as inaccessible
rather than quiet. The forum was readable, but its newest posts were announcements — a call for
papers 18 hours old and a summit page 18 days old — and its newest technical post was seven days
old; two of those posts are summarised at the end for what they say about where users get stuck.

### The answered series, and the rule that does the real work

The interesting part of the modify-return series is not the scan but the exception, because the scan
is a data-flow trick and the exception is a statement about kernel programming. The failure it exists
for is a function that releases a lock on its error path and then returns an error: a caller that
receives that error from a skip finds the lock held, and the next unlock underflows or the next
lock acquisition deadlocks. The tool resolves it from machine code alone — a `lea` of a member of an
argument, a call to a function that BTF says returns void and takes a pointer, then the constant
load — with no source annotation and no name list, which is the difference between a rule a reviewer
can audit and a list that rots. The open question is what remains after the exception: the author's
own runs leave about 114 damaging functions for a future name-based deny list, and the stated blind
spot of the exception is cleanup performed by a value-returning helper, where a skip really can look
like a failure.

### Grace periods that nobody needed

A second thread argued that the BPF link aggregator takes a full RCU grace period on every attach,
including the first attach to a freshly created device, and that the wait sits inside the network
configuration lock, so bulk container creation serialises the whole machine's control plane behind
two grace periods per device. The reviewer's reply is a useful correction of the premise: the wait
protects the previous entry, not the new one, so the fix is not to remove it but to borrow the
scheme the queue-discipline counterpart in the same hook already uses — start a grace period after
the swap, and wait before the old buffer is written again; a new bundle starts from a
before-completion marker, so the first attach waits for nothing. The resulting table claims the wait
disappears for the first attach and for a second program after a queue discipline was attached
first, and shrinks to "only when the previous change is younger than a grace period" otherwise. The
lesson generalises: when a series claims a wait is unnecessary, check which generation of the object
the wait protects.

### A per-subsystem mutex, argued with the numbers that justify it

An applied series moved cgroup BPF state off the global cgroup mutex. The motivating evidence is
worth keeping as a pattern for how to argue a lock split: per-day hung-task counts from one fleet,
over a thousand hosts waiting in the program-query path, dozens in link information, hundreds in
release work, with the holder almost always a directory create or remove stuck on a sibling lock. The
split gives the BPF state its own mutex, keeps attach, detach and replace under the global lock
because they walk descendants, and fixes two ordering defects the new locking made visible: reading
an effective array that had already been freed, and attaching to a cgroup that was being removed.
When a performance series arrives with per-host counts of the actual stall, the review conversation
tends to be about correctness rather than about whether the problem is real.

### A context field that reports one width and produces another

An applied fix closed a verifier model mismatch on a socket-buffer context field. The conversion for
the remote port loads the 16-bit member and, on little endian, shifts it into the high half of a
32-bit register; it reported its width through the generic helper as the member's two bytes, so a
two-byte read was classified as a narrower access of a two-byte field and the masking step that
widens a narrow read by a shift and a mask was skipped. The verifier then believes the value is at
most `0xffff` while the register holds `0xffff0000` plus the port, which turns arithmetic the
verifier folds into a constant into an out-of-bounds access relative to a map value. The fix is one
line of intent: report the width the conversion actually produces. The habit worth keeping is to
compare the reported width against the instructions the conversion emits, because the two live in
different files and nothing checks that they agree.

### When the patch is right and the bug is not there

Two threads in the window have the same shape at different sizes. A proposed mask of an absolute
packet load's address, argued from a transient out-of-range offset potentially dereferencing a
secret, was rejected with the note that the conversion of a classic absolute load inlines its own
bounds check in the fast path before the helper fallback, so the patch changes nothing observable.
The second was the narrower version of the same disagreement, where the reporter's evidence came
from an instrumented kernel with a cache flush inserted in front of the branch, and the reviewer's
answer was that nobody runs such a kernel. Both are a reminder that the object under review is the
converted program, not the program the author wrote.

### Callchains that start with the same frame twice

A three-patch series fixes a duplicate caller frame in arm64 kernel callchains and adds selftests for
the paths that build them. The mechanism: the helper that captures caller registers sets the program
counter to the caller's return address but points the frame-record register at the current
function's own record, whose saved link register is that same return address; the unwinder then
reports the program counter and unwinds into the duplicate. The fix uses the caller's frame record,
which is what the unwinder's own initialisation already does, and the same defect was fixed on
x86-64 years earlier. It matters to BPF users directly, because stack helpers in tracepoint, raw
tracepoint, typed tracepoint and fentry programs all use that capture path, so the duplicate appears
at the top of collected stacks rather than in a corner of perf.

### Documentation that drifted away from the verifier

Two documentation patches and one encoder fix round out the window. The documentation patches record
that a socket-map mutation restriction has existed in the verifier for years and was never written
down: only iterator tracing, sock-ops for deletion, reuseport and socket-lookup programs may update
or delete, and everything else is rejected with a named error. The encoder fix is a direct follow-up
to a defect reported earlier in the week: functions marked as kernel functions are now exempt from
the location-list checks that used to drop them from BTF, and are collected whenever BTF is encoded
instead of only under one feature flag, which brings two lost functions back with no other change to
the blob. Both are the cheap kind of fix: the gap was between the code and the words, or between two
stages of the same build.

### What practitioners outside the kernel list are asking

The forum's recent technical posts show two recurring shapes. First, enforcement without a
signature: a project intercepts file, network and exec operations for an unprivileged process,
learns a baseline of normal behaviour, and then has to decide how to react without taking the host
down — it refuses to touch the SSH session or the service manager for exactly that reason. Second,
ramp-up: a graduate-level question asks for a path from tracing tools to a from-scratch program
against a stealthy kernel-side implant, and the useful part of the question is the recognition that
verifier and type-format failures, not the hook mechanics, are the time sink. Neither produced a
question narrow enough to answer here, and neither carries an identity or deployment detail that a
reader needs.