# Can an unprivileged container create its own BPF token?

**Short answer:** yes, but only after a privileged manager has created the authority from which the token is derived. A process that is unprivileged in the host's initial user namespace can call `BPF_TOKEN_CREATE` itself when all of the following are true:

- it runs in the same non-initial user namespace that owns a dedicated bpffs instance;
- that bpffs instance was configured with explicit `delegate_cmds`, `delegate_maps`, `delegate_progs`, and/or `delegate_attachs` masks; and
- the caller has `CAP_BPF` in that user namespace.

The resulting file descriptor is not a bearer credential that replaces capabilities. Later map, BTF, and program operations need both the token and the capabilities required for the operation in the token's user namespace. The token changes where capability checks are evaluated and bounds what may be requested; it does not manufacture privilege.

This means an orchestrator can let a workload derive its own token, but the workload cannot create the delegation policy from nothing or enlarge it. The clean division is: the privileged manager defines a least-privilege bpffs delegation, while the workload derives and consumes a token inside that boundary.

## There are two different acts: delegate and derive

The confusing part of the design is that “create a token” sounds like one operation. It is actually the second half of a two-stage protocol.

First, a privileged manager prepares a bpffs instance owned by the workload's user namespace. It configures four independent bit masks on that filesystem:

- `delegate_cmds` limits which `bpf()` commands may use the token;
- `delegate_maps` limits map types available through an allowed `BPF_MAP_CREATE`;
- `delegate_progs` limits program types available through an allowed `BPF_PROG_LOAD`; and
- `delegate_attachs` limits expected attach types for an allowed program load.

The kernel's bpffs implementation accepts those as filesystem parameters and stores them in the superblock. The token later copies those masks. A workload that receives the prepared mount cannot rewrite the delegation masks merely because it owns the associated user namespace; the upstream selftest explicitly checks that an unprivileged child gets `EPERM` when it tries.

Second, a process in that user namespace opens the **root** of the delegated bpffs and calls `BPF_TOKEN_CREATE`. The kernel verifies the filesystem, namespace, capability, and delegation state, then returns an anonymous token file descriptor. The workload has derived a concrete handle from a policy that somebody more privileged established earlier.

The manager may instead derive the token and pass its file descriptor over a Unix-domain socket with `SCM_RIGHTS`. The upstream selftest exercises file-descriptor passing. Both arrangements preserve the same authority boundary: possession of the descriptor does not let the receiver modify its masks or skip local capability checks.

## The minimum low-level handshake

With a recent libbpf, the workload-side derivation is intentionally small:

```c
int bpffs_fd = open("/run/workload-bpf", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
if (bpffs_fd < 0)
    /* report errno */;

int token_fd = bpf_token_create(bpffs_fd, NULL);
if (token_fd < 0)
    /* report errno */;
```

The direct syscall form sets `attr.token_create.bpffs_fd` and invokes `bpf(BPF_TOKEN_CREATE, ...)`. `bpffs_fd` must refer to the filesystem root, not a pinned object or a subdirectory.

Downstream operations have their own token fields. For example, a low-level map creation uses `bpf_map_create_opts.token_fd` and sets `BPF_F_TOKEN_FD` in `map_flags`; program and BTF loads do the corresponding thing in their option flags. Setting the descriptor without the flag, or the flag without a valid token descriptor, is not the handshake.

Applications that load a normal BPF ELF object can avoid threading the descriptor through every call. `bpf_object_open_opts.bpf_token_path` tells libbpf which bpffs root to use. If it is not set, libbpf consults `LIBBPF_BPF_TOKEN_PATH` and otherwise attempts the default `/sys/fs/bpf`. An empty value disables automatic token creation. When enabled, libbpf derives the token and applies it to supported map creation, BTF loading, and program loading operations for that object.

## A token and namespaced capabilities are both required

The upstream selftest checks the four combinations for a privileged map creation:

| Token supplied | `CAP_BPF` in token user namespace | Result |
| --- | --- | --- |
| No | No | Failure |
| Yes | No | Failure |
| No | Yes | Failure |
| Yes | Yes | Success, if the command and map type were delegated |

Program loading can require more than `CAP_BPF`. The kernel UAPI names `CAP_BPF`, `CAP_PERFMON`, `CAP_NET_ADMIN`, and `CAP_SYS_ADMIN` as capabilities whose checks may be evaluated in the token's user namespace. The exact set depends on the program type, helpers, and operation. The kernel selftest deliberately loads an XDP program using helpers that exercise `CAP_NET_ADMIN`, `CAP_BPF`, and `CAP_PERFMON` checks.

This is why “unprivileged container” needs a precise definition. The workload may have no relevant capability in the host's initial user namespace while still holding narrowly scoped capabilities in its own user namespace. BPF tokens make that useful to selected BPF operations. A process with no required capability anywhere does not gain one merely by receiving a token FD.

## Read `errno` as a failed precondition

For `BPF_TOKEN_CREATE` itself, the current kernel implementation exposes a useful failure map:

- `EBADF`: the supplied bpffs descriptor is invalid;
- `EINVAL`: the descriptor is not the root of a bpffs instance;
- `EPERM`: the caller is not in the bpffs owning user namespace or lacks `CAP_BPF` there;
- `EOPNOTSUPP`: creation was attempted in the initial user namespace, where a token would not relocate capability checks; and
- `ENOENT`: the bpffs instance has no delegation mask configured.

A filesystem access check can also return an ordinary path-permission error, and a security module can reject token creation or use.

If token creation succeeds but a later BPF operation fails with `EPERM` or `EINVAL`, inspect a different set of facts:

1. Was `BPF_F_TOKEN_FD` set in the correct flags field, and was the right token FD supplied?
2. Does the token allow the requested `bpf()` command, map type, program type, and expected attach type?
3. Does the process still hold every required capability in the token's owning user namespace?
4. Did the verifier reject the program, or did an LSM or target-specific attach check deny it?

`BPF_OBJ_GET_INFO_BY_FD` can return a token's four allowed masks, and `/proc/self/fdinfo/<token-fd>` renders them for diagnostics. The bpffs mount options are also visible through normal mount information. Log these derived facts and error codes in a controlled diagnostic environment; do not treat a token FD number as meaningful evidence because descriptor numbers are process-local and reusable.

## What the boundary does not authorize

A BPF token is deliberately narrower than “BPF inside a container is now unrestricted”:

- The token inherits the bpffs delegation masks. `BPF_TOKEN_CREATE` does not currently accept a second, per-token mask that the deriving workload can widen or narrow.
- The verifier still checks program safety, helper availability, context access, and all normal program rules.
- LSM token hooks and operation-specific security hooks still run.
- Target ownership and attachment rules still apply. Permission to load a program is not automatically permission to attach it to an arbitrary host object.
- Closing the last reference invalidates the handle; passing or duplicating the FD preserves the same token, not a new policy.

For multi-tenant systems, use a separate delegated bpffs instance per trust boundary, delegate only the commands and types that the workload actually needs, and avoid exposing the host's general-purpose bpffs mount. Treat the user namespace, the mount configuration, the capability set, and the token FD as one security design. Auditing only one of the four gives a misleading answer.

## A practical validation sequence

Before putting the mechanism into an orchestrator, reproduce the upstream matrix with the smallest harmless object:

1. Confirm that token creation fails on a normal directory, a bpffs subdirectory, an undelegated bpffs, and the initial user namespace.
2. In the workload user namespace, confirm that token creation succeeds only with local `CAP_BPF`.
3. Pick one delegated map type. Prove that token plus capability succeeds, while token-only, capability-only, and neither all fail.
4. Request one map or program type outside the masks and prove it is denied.
5. Repeat for every program and attach type the production workload needs, including its additional capabilities and target permissions.
6. If the token or bpffs FD crosses a process boundary, test the exact `SCM_RIGHTS` or inherited-FD path and close unexpected copies.

This sequence proves the security boundary, not merely that one happy-path loader happened to work.

## References

- [Linux BPF UAPI: `BPF_TOKEN_CREATE` semantics and user-namespace capability checks](https://github.com/libbpf/libbpf/blob/master/include/uapi/linux/bpf.h#L3859-L3912)
- [Linux kernel token implementation: creation checks, inherited masks, fdinfo, and security hooks](https://github.com/torvalds/linux/blob/master/kernel/bpf/token.c)
- [Linux bpffs implementation: delegation mount parameters and rendered masks](https://github.com/torvalds/linux/blob/master/kernel/bpf/inode.c)
- [Linux BPF selftest: delegated bpffs setup, FD passing, capability matrix, and denied reconfiguration](https://github.com/torvalds/linux/blob/master/tools/testing/selftests/bpf/prog_tests/token.c)
- [libbpf low-level BPF API, including `bpf_token_create`](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/bpf.h)
- [libbpf object-open token path and `LIBBPF_BPF_TOKEN_PATH`](https://github.com/libbpf/libbpf/blob/master/src/libbpf.h#L2540-L2575)
- [OpenTelemetry GenAI contributor guidance for sanitized VCR cassettes and live re-recording](https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/AGENTS.md)
- [OpenTelemetry declarative configuration for semantic-convention stability opt-in](https://github.com/open-telemetry/opentelemetry-configuration/blob/main/opentelemetry_configuration.json)
- [Linux BPF discussion: KASAN coverage for BPF JIT memory](https://lore.kernel.org/bpf/20260828-kasan-v8-0-7c1c0fdb9d7f@bootlin.com/T/#t)
- [Linux BPF discussion: avoiding a null dereference while dumping a key-less BTF map](https://lore.kernel.org/bpf/8b3e7f24-795d-458b-a24e-fe154b0cf03d@linux.dev/T/#t)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected question came from the 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, workspace and channel identities, message links, exact times, private topology, raw logs, and searchable wording have been removed. No raw transcript was retained, and no social interaction was performed.

### Delegation needed a failure-oriented explanation

The strongest unresolved question asked whether a workload could perform the entire BPF token handshake itself and why seemingly reasonable attempts failed. The missing distinction was between configuring delegation and deriving a token. Public kernel code and selftests show that the workload can own the latter step, but only after a privileged manager has created a compatible user namespace and delegated bpffs. The four-way token/capability matrix and the `errno` map above make the trust boundary testable instead of treating successful token creation as sufficient proof.

### Provider tests must separate reproducibility from live proof

An instrumentation discussion returned to a practical contributor problem: how to test provider-specific changes when a live service credential is unavailable. The public contributor guidance supports a layered answer. Deterministic request and response behavior belongs in sanitized VCR cassette or unit tests; authentication headers, cookies, organization identifiers, and account-identifying body fields must be removed. An AI-synthesized cassette should be marked as temporary and later re-recorded against the real provider. Such a fixture can validate code paths, but it is not evidence that today's provider endpoint and credentials work.

Related implementation work was moving multiple provider integrations toward shared GenAI utilities. That increases the value of a common conformance suite: provider adapters should prove the same span structure and redaction invariants, while a small, controlled live-check layer covers SDK serialization and service drift.

### Semantic-convention switches need a retirement plan

Another active theme concerned feature flags for breaking telemetry-schema changes. A flag per transition can make a rollout reversible, and the common stability opt-in supports phased migration. But a growing list of permanent flags becomes a second schema. The durable design is to define the old, duplicate-emission, and new states; test each state; record precedence when global and domain-specific configuration coexist; and assign a removal condition once downstream consumers have migrated.

### Kernel work focused on making failure states observable

The public kernel discussion was active around KASAN visibility for JIT-generated memory, a null dereference while dumping a map without a key, verifier precision around callback registers, indirect trampoline tests, and profiler reporting. The common concern was observability at subsystem boundaries: distinguish JIT memory corruption from verifier-state loss and from userspace-tool assumptions, then preserve a minimal selftest for the specific boundary that failed.

Several project-specific chat surfaces were quiet in the daily window, while other visible activity consisted of review, merge-order, or conflict-resolution requests rather than new technical questions. Those targets were still counted as accessible and checked; they were not converted into invented issues or reused as evidence for the selected answer.
