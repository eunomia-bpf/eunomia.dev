# Why can classic uprobe BPF programs crash preemptible kernels that use private BPF stacks?

**Short answer:** a classic uprobe invocation can stay on one CPU while still being preempted. On x86 kernels that give an eligible BPF program one private stack per program and CPU, a second task can then run the same program on that CPU and reuse the first invocation's stack. When the first task resumes, a spilled register that the verifier proved to be a valid pointer may contain unrelated runtime data. A direct load or map helper can consequently fault inside JITed code and panic the kernel.

This is best treated as an execution-invariant bug, not evidence that the verifier accepted an invalid pointer operation. The verifier checked one abstract execution in which spills are restored intact. The failure happens later because two concrete executions can share storage that was designed under a same-CPU exclusivity assumption.

The issue is also narrower than “all uprobes are unsafe.” The private-stack eligibility and x86 JIT support entered the Linux 6.13 development cycle. The affected path therefore depends on the exact kernel build, architecture, JIT implementation, program stack use, uprobe attachment path, and preemption behavior. Vendor backports make version-only rules unreliable.

## How the corruption can happen

Private BPF stacks reduce pressure on the kernel stack. The x86 JIT support allocates the private storage per subprogram and CPU, and the eligibility logic requires runtime recursion protection because only one copy exists for each CPU. That design is safe only if two invocations of the same program cannot overlap while using that copy.

Classic uprobes violate that assumption in the path under investigation:

1. Task A enters a classic uprobe program on CPU 0 and begins using that program's CPU-0 private stack.
2. The uprobe runner uses `migrate_disable()`, so Task A cannot migrate to another CPU, but this does not by itself prevent preemption.
3. Task A is preempted while registers or verifier-tracked pointers are spilled on the private stack.
4. Task B runs on CPU 0, hits the same uprobe, and invokes the same BPF program.
5. Task B reuses and overwrites the same per-program, per-CPU private-stack slots.
6. Task A resumes and reloads a corrupted spill. Native code now operates on a value that no longer matches the pointer type proved by the verifier.

The visible crash site can therefore be misleading. It may appear in a BPF map helper, a normal map-value load, or another dereference after the bad restore. Adding a null check for one application pointer may fix an independent program bug, but it does not prove that private-stack corruption is gone.

One public reproducer made that distinction concrete: an application-level missing null check needed repair regardless, while JITed execution also restored a value inconsistent with the verified register state. A separate public report reproduced a panic under concurrent TLS and process activity, with KASAN locating the invalid access in a BPF program reached through a uprobe. These observations support the private-stack hypothesis, but the proposed kernel fix had not yet been accepted upstream at the time of this review.

## Confirm the mechanism before changing production kernels

Collect enough evidence to distinguish this bug from an ordinary bad offset, stale userspace ABI, or application error:

```bash
uname -a
grep -E 'CONFIG_(BPF_JIT|PREEMPT|PREEMPT_DYNAMIC)=' /boot/config-"$(uname -r)"
bpftool prog show
bpftool prog dump xlated id PROG_ID opcodes linum
bpftool prog dump jited id PROG_ID opcodes linum
```

The last command requires JITed instructions to be available to the caller. Preserve the matching BPF object, BTF, program ID/name, attach type, translated instructions, JIT dump, and exact kernel build. Keep raw panic output in the incident system, not in a public issue or content repository unless it has been reviewed for secrets and private topology.

Then reduce the workload along independent axes:

- Disable only the suspected classic uprobes while leaving unrelated BPF programs loaded. If the panic disappears, the attachment set is narrowed without blaming every BPF workload.
- Reduce to one program and one symbol, then increase concurrent invocations on one CPU. The suspected failure needs overlapping executions of the same program on the same CPU; ordinary single-shot tests may never trigger it.
- Compare an exact kernel build containing the private-stack commits with a build that does not contain them, or with a build carrying a candidate kernel fix. Check commits, not just release strings.
- Test the program's own pointer and offset handling separately on an unaffected kernel. Program bugs and a kernel execution bug can coexist.
- Compare translated instructions with the native faulting instruction. The key signal is a runtime register value inconsistent with the verifier-approved type, not merely a fault inside a helper.

A PREEMPT configuration is a risk clue, not a complete diagnosis. Conversely, failure to reproduce on a lightly loaded system does not clear the path; the interleaving window may simply be rare.

## Choose mitigations by which invariant they restore

The safest immediate mitigation is to detach or disable the affected classic uprobe programs on kernels whose behavior has not been validated. If instrumentation is optional, losing that signal is preferable to risking host availability.

For kernel selection, use a build that either does not activate this private-stack path or contains an accepted fix and has passed the concurrent reproducer. A nominal downgrade to a particular release is not sufficient when a distribution may backport the private-stack changes. Likewise, do not disable the BPF JIT globally as an improvised production workaround without a separate compatibility, performance, and security review.

Two candidate fixes illustrate the correct invariants but remain proposals:

- A kernel-side patch acquires the existing per-program recursion context around each real program in the classic uprobe array. If the same program is already active on that CPU, it increments the missed-invocation counter and skips the nested run. This restores the single-user assumption for the per-CPU private stack.
- A draft application patch wraps every uprobe-capable program body with the weak `bpf_preempt_disable()` and `bpf_preempt_enable()` kfuncs. Tail calls require special handling because the verifier forbids returning or tail-calling with preemption left disabled. This keeps the first invocation from being preempted, but only if every entry, exit, and tail-call path is covered.

Neither proposal should be described as an upstream fix yet. The kernel-side approach centralizes the invariant and protects applications without requiring each BPF project to know about the implementation detail. The application guard can be a bounded bridge for a project that controls all relevant programs, but its verifier compatibility and runtime cost must be tested across every supported kernel.

After applying any mitigation, repeat the concurrent reproducer, inspect missed-run accounting if the kernel fix exposes it, and run ordinary functional and load tests. “No panic once” is not enough: also verify that the intended probes still attach, expected events are not silently lost, and tail-call paths load on the oldest supported kernel.

## References

- [Linux commit: select BPF subprograms eligible for private stacks](https://github.com/torvalds/linux/commit/a76ab5731e32d50ff5b1ae97e9dc4b23f41c23f5)
- [Linux commit: add x86 JIT support for per-subprogram, per-CPU private stacks](https://github.com/torvalds/linux/commit/7d1cd70d4b16ff0216a5f6c2ae7d0fa9fa978c07)
- [Linux commit: implement the uprobe program-array path used by sleepable uprobes](https://github.com/torvalds/linux/commit/8c7dcb84e3b744b2b70baa7a44a9b1881c33a9c9)
- [Public kernel-panic report and reproducer references](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/3056)
- [Draft application guard using BPF preemption kfuncs](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/pull/3059)
- [Experimental kernel patch: protect classic uprobes with per-program recursion context](https://github.com/simonswine/beyla/blob/20c9912a02c5d5924c992faf7ee353c7d71c9fa0/evidence/kernel-6.18-bpf-3733/0001-bpf-guard-classic-uprobes-against-private-stack-corr.patch)
- [bpftool program inspection and JIT-dump documentation](https://github.com/libbpf/bpftool/blob/main/docs/bpftool-prog.rst)
- [BPF mailing-list proposal: folio-backed scratch and pool allocators](https://lore.kernel.org/bpf/20260817-folio-pool-v1-v1-0-0c1d230aa3af%40gmail.com/T/#t)
- [BPF mailing-list proposal: aggregate return values up to 16 bytes](https://lore.kernel.org/bpf/20260817042141.2286086-1-yonghong.song%40linux.dev/T/#t)
- [Prempti architecture: tool-call hooks evaluated by Falco rules](https://prempti.falco.org/)
- [ActPlane architecture: kernel-level policy enforcement for agent process trees](https://github.com/eunomia-bpf/ActPlane)
- [Crash-safe eBPF dataplane loader design and implementation](https://erwinkok.org/posts/ebpf-dataplane-loader/)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The public forum was reviewed through its ordinary visible legacy interface. The selected question came from the 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, channel identities, message links, exact times, private topology, original logs, and searchable phrasing have been removed. No raw transcript was retained.

### A verifier proof still depends on the runtime preserving its machine state

The strongest same-day discussion connected a host panic to classic uprobe execution on a preemptible, private-stack kernel. The practical lesson is that verifier acceptance proves properties of the BPF instruction model; the JIT and invocation path must preserve the registers and stack slots that make that proof true. When the runtime restores a different value into a verifier-tracked pointer register, a safe-looking map access can become a native fault.

The investigation also showed why two bugs must not be collapsed into one. A probe can mishandle a legitimate application-level null value, while an overlapping kernel invocation independently corrupts a saved register. Fix both, test each on an isolated kernel path, and do not treat a passing null-check test as evidence for a repaired kernel invariant. The public kernel and application patches were still proposals, so the conservative operational answer remains selective probe disablement or a validated kernel build.

### Agent guardrails need explicit coverage and failure semantics

A second discussion compared two runtime-enforcement layers for coding agents. One evaluates structured tool requests before execution and returns allow, deny, or ask verdicts through a rule engine. This gives precise intent-level policy and a useful audit story, but it cannot see syscalls performed inside an allowed shell command. The other follows process lineage and enforces file, execution, network, information-flow, and causal-order rules in the kernel. It covers indirect subprocess behavior, but requires Linux capabilities, kernel support, and careful definitions for block, kill, notify, and fail-closed behavior.

These layers are complementary, not interchangeable. A review should state the observation point, the action point, how state persists across events, what happens when the policy engine is unavailable, and which bypasses remain. Tool hooks provide high-level context; OS enforcement provides effect-level coverage; isolation still supplies a final resource boundary.

### New BPF capabilities are being designed across allocator, ABI, and verifier boundaries

The public development list was especially active around two proposals. A folio-backed bump allocator aims to replace repeated small allocations in batch-shaped paths such as verifier stack-state nodes and generic map-update buffers. The claimed advantage is cheap linear allocation and bulk teardown, but the lifecycle must really be region-shaped: individual ownership, partial frees, reclaim pressure, and fallback objects need explicit handling. Because the series is an initial proposal with runtime switches and subsystem-specific adoption, it should be benchmarked and fault-injected rather than treated as an available kernel API.

Another series extends kfunc and BPF-subprogram returns to aggregates up to 16 bytes in the `R0:R2` pair. This is not merely an ABI change. JITs must place the second half correctly, precision backtracking and liveness must model `R2`, the verifier must prevent pointer provenance from being laundered through aggregate fields, and interpreter fallback must not silently use different semantics. The proposed constraints—scalar-only aggregate members for globally checked boundaries, architecture opt-in for kfunc returns, and JIT-required execution—show the kind of end-to-end contract a wider return ABI needs.

### Crash-safe loaders should reconcile kernel state instead of assuming process lifetime

The public forum surfaced a new loader design focused on recovery after the userspace process exits unexpectedly. The useful pattern is to treat pinned objects and links as observed kernel state, keep a separate desired configuration, and make startup reconciliation idempotent. Every object needs an owner, a version or generation, and a terminal cleanup rule; otherwise a restarted loader cannot tell reusable state from debris.

That design also separates blocking kernel operations from the async control plane and tests decision logic without requiring a live kernel for every case. There was not yet a substantive follow-up discussion on the post, so it is best read as an implementation proposal rather than community consensus. The remaining chat and project-specific surfaces were quiet, contained automated notifications, or had no new technical question in the daily window.
