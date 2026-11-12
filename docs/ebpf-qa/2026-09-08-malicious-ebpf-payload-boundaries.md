# Which technical boundaries actually constrain malicious eBPF payloads in practice?

**Short answer:** none of them is what most people assume. The verifier is a *memory-safety* check, not a *behavioral-safety* check: it proves a program cannot corrupt kernel memory (barring a verifier bug), and it says nothing about whether the program has malicious intent. A malicious payload is constrained in practice by four separate, testable boundaries, each with a known failure mode:

1. **The load gate** — which principals may attach programs at all (capabilities, the unprivileged-BPF sysctl, user namespaces).
2. **The verifier's memory-safety scope** — what a loaded program may do to memory; a guarantee that moves with every kernel release and differs across architectures in practice.
3. **The portability boundary (BTF/CO-RE)** — payloads that assume specific kernel struct layouts only work where BTF exists and matches.
4. **The visibility boundary** — loaded programs are invisible to process listings, syslog, and auditd; they are observable only through BPF-native telemetry.

If you are scoping an empirical study (or building a blue-team detection baseline), these four are the boundaries to map. "The verifier blocks aggressive state manipulation" is only boundary 2, and it is the narrowest of the four: a payload that passes the verifier has already cleared the only static gate. Everything after that is capability, portability, and observability — not static analysis.

## Why "verifier-approved" is not "safe"

The kernel's trust model treats the verifier as the single static gate: programs are untrusted bytecode until verified, then JIT-compiled and executed, and the helpers they call do not re-check memory safety. The chain of trust from a verified program to the kernel is only as strong as the verifier itself. A 2025 systematic analysis (SoK, USENIX Security) of eBPF memory safety classifies the verifier's weaknesses into three families, each demonstrated with real CVEs:

- **Unsound checks** — optimizations such as path pruning rely on exact register-dependency tracking; miss one dependency and an unsafe path can be pruned as if it were safe (CVE-2023-2163).
- **Incomplete checks** — a helper whose prototype has no size argument skips size validation entirely (CVE-2021-4204, via `bpf_ringbuf_submit`); the verifier trusts helpers, and helpers do not re-validate.
- **Limited scope** — the checks stop at the bytecode; temporal-safety issues inside helpers (for example use-before-initialization through `dev_map_lookup_elem`) are invisible to the verifier.

The same study found that 1.62–3.74% (37–85) of the memory operations in *public* eBPF programs cannot be proven comprehensively memory-safe — that is ordinary tooling, not malicious code, and it is what verification currently leaves unproven. Add a sustained CVE stream through the verifier (tens per year in recent years, hundreds of fuzzer-reported memory-error bugs since eBPF shipped in kernel 4.4), and the practical conclusion is: the memory-safety boundary moves with every kernel release. A program rejected on one version may load on another.

None of this means the verifier is weak in a day-to-day sense — for a *legitimate* program, memory corruption is the actual risk and the verifier exists to prevent it. It means "verifier-approved" is a narrower property than "safe," and defenders should not conflate the two.

## Boundary 1: the load gate

What constrains *who* can load programs is capability policy, not the verifier:

- Unprivileged BPF loading is disabled on modern distributions (`kernel.unprivileged_bpf_disabled=1`). That sysctl is the door that historical local privilege escalations came through: in the CVE-2016-4557 / CVE-2018-18445 era, unprivileged bytecode injection and verifier bounds bypasses were the entry point.
- Since Linux 5.8, `CAP_BPF` separates BPF load/attach from `CAP_SYS_ADMIN`, so a service holding only `CAP_SYS_ADMIN` no longer automatically holds BPF, and vice versa.
- BPF LSM, where enabled, can gate load/attach/pin on MAC policy.

The empirical pattern across the public eBPF rootkits (from the Black Hat USA 2021 "With Friends Like eBPF" work and its open-source descendants) and the first documented production eBPF backdoor is the same: **these are persistence and stealth techniques, not entry techniques.** The actor already holds root or `CAP_SYS_ADMIN`; eBPF is used to stay hidden, persist through pinned maps, and route commands through the data path — all inside verifier-approved behavior. The load gate answers "who," not "what they do after."

## Boundary 2: verifier scope, per version and architecture

For an x86_64 vs ARM64 matrix, be precise about what actually differs:

- The verifier is shared C code; it is not "stricter" on ARM64 than x86_64 for the same program.
- The JITs are per-architecture, and architecture-specific bugs historically appear in JIT paths and in per-architecture selftest coverage. An active 2026 bpf mailing-list series is exactly about this: a buffer overflow in the powerpc JIT for large BPF programs plus a patch explicitly aimed at *missing verifier selftest coverage* on that architecture. That kind of per-architecture verification and test debt is precisely what a cross-architecture study should measure.
- Kernel configuration differences (enabled BPF features, `CONFIG_*` options, BTF availability) change what can load on two kernels of the same version.

So the unit of measurement is not "architecture" or "version" alone but the tuple **(architecture, kernel version, kernel configuration, BTF availability)**. A payload that loads on x86_64 with full BTF may fail on ARM64 of the same version — or load, then be rejected by an architecture-specific check. Both outcomes are boundary data.

## Boundary 3: the portability boundary (BTF/CO-RE)

CO-RE is the boundary that makes payloads *less* portable, not more. A program built against one kernel's BTF carries relocations that must resolve against the target kernel's BTF:

- No BTF on the target kernel (stripped or never built) → CO-RE programs cannot load at all.
- Struct-layout drift between source and target → relocations fail, or silently produce wrong field offsets where the types are compatible but shifted.
- The kernel ABI itself is moving: a large patchset in review extends BPF with by-value struct and `__int128` arguments (including for kfuncs), so what a payload can express changes between releases.

For a defensive study this is also a *detection* boundary: load failures on BTF mismatch leave a signature in the load path (`BPF_PROG_LOAD` failing on relocation), which a load monitor can flag. For an offensive study it is the *portability ceiling*: hand-tuned, non-CO-RE offsets are the only way to dodge BTF requirements, and that path is far less portable and more likely to trip verifier bounds checks.

## Boundary 4: the visibility boundary

This is the boundary where most defenses actually fail, and the one recent public research keeps returning to:

- eBPF programs do not appear in `/proc`, and their activity does not appear in syslog or auditd. A compromised host can run hostile kernel logic while its usual telemetry looks unremarkable.
- Pinned maps under `/sys/fs/bpf` provide persistence across program unload.
- The first documented production eBPF backdoor (BPFDoor) operated entirely within verifier-approved behavior: a data-path hook as the command channel, pinned maps for persistence, nothing in the process table.
- Threat-intelligence research now classifies eBPF implants as a distinct Linux malware class, explicitly because they operate inside verifier-approved boundaries.

The only reliable visibility is BPF-native: `bpftool prog show` / `bpftool map show`, listing `/sys/fs/bpf`, and runtime monitors (Falco rules, Tracee, eBPFmon-style tooling) that watch `BPF_PROG_LOAD` / `BPF_MAP_CREATE` / attach events. If a detection stack cannot correlate BPF load/attach events with identity and intent, boundary 4 is open and the other three matter less.

## How to measure the boundaries (a practical experimental design)

If the goal is a reproducible technical matrix, structure it as five independent axes, each testable with public tooling:

1. **Load-gate matrix** — principals (unprivileged, `CAP_BPF` only, `CAP_SYS_ADMIN` only, root, user-namespace + `CAP_BPF`) × distro/kernel configurations, recording `BPF_PROG_LOAD` acceptance for an identical minimal program.
2. **Verifier boundary per (arch, version, config)** — run the public CVE reproducer set (fuzzer reproducers, `selftests/bpf`) on each target; record which verifiers reject, accept, or crash. This is where "verifier pushback" is actually measurable.
3. **CO-RE portability** — build one CO-RE program against N kernel BTF sets; count successful vs failed loads per target, including at least one target with BTF disabled.
4. **Performance cost** — benchmark syscall-interception and map-heavy workloads per architecture; this is simultaneously the "cost of evasion" number and the detection-signal number, because the same baseline flags anomalies.
5. **Visibility gap** — for the same workload, compare what `/proc` + syslog + auditd show against what `bpftool` shows, and quantify the difference.

Axes 1, 4, and 5 need no offensive payload at all: they use legitimate programs and measure the environment. Axis 2 uses published reproducer programs that have been public for years. That keeps the work in defensive-research territory while producing the matrix blue teams actually lack.

## Where the answer stops applying

- These are **Linux kernel** boundaries. eBPF on other runtimes (user-space interpreters, Wasm-BPF, non-Linux ports) has a different and often weaker set of guarantees.
- The matrix is only as fresh as the kernels it was measured on: verifier behavior, distribution defaults, and the capability split all change between releases, so re-measure instead of quoting.
- "Passes the verifier" is a memory-safety statement, not a safety statement. A verified program can still exfiltrate kernel data through maps, attach to privileged hooks, or hide — which is exactly why boundary 4 exists.
- The historical unprivileged-BPF attack era is largely closed on current distributions (unprivileged loading off by default); a modern threat model should assume post-root persistence and stealth, not unprivileged entry.

## References

- [Linux kernel documentation: eBPF verifier](https://docs.kernel.org/bpf/verifier.html)
- Huang et al., "SoK: Challenges and Paths Toward Memory Safety for eBPF", USENIX Security 2025 — [paper](http://www.nebelwelt.net/files/25Oakland.pdf)
- Linux Security, "Linux eBPF Security Advisory: Critical Visibility Concerns 2025:0011-2" — [article](https://linuxsecurity.com/features/ebpf-abuse-linux-kernel-visibility-gap)
- "With Friends Like eBPF, Who Needs Enemies?", Black Hat USA 2021 — [slides](https://i.blackhat.com/USA21/Wednesday-Handouts/us-21-With-Friends-Like-EBPF-Who-Needs-Enemies.pdf)
- Red Canary, "eBPF Malware" threat-detection analysis — [article](https://redcanary.com/blog/threat-detection/ebpf-malware/)
- eBPF Foundation, "Threat model and independent verifier audit" — [post](https://ebpf.foundation/threat-model-and-independent-verifier-audit-examine-the-security-of-ebpf/)
- eBPF.io, "What is eBPF?" (kernel hardening measures) — [post](https://ebpf.io/what-is-ebpf/)

## Community discussion today

Coverage this run: one of the two archive-opted-in Slack workspaces was readable — one public channel, zero messages in the strict 24-hour window and seven messages in the seven-day fallback window. The other opted-in workspace's two allowlisted channels were not present in its archive, recorded as inaccessible rather than quiet. The two allowlisted Discord workspaces are visible-browser-only surfaces and no visible browser session was available for them this run, reported here as a coverage gap rather than silence. The public bpf@vger.kernel.org archive and the public practitioner forum were reviewed through their ordinary public pages.

### A recurring question: where does malicious eBPF actually hit a wall?

The strongest discussion in the window was a research-scoping exchange, present in both the readable archive channel and the public practitioner forum, asking whether an empirical study of the technical boundaries that restrict malicious eBPF payloads — verifier pushback, CO-RE limitations, and the performance cost of kernel manipulation across x86_64 and ARM64 — would be useful to the defensive community, and where to start. The published answer above is the response to it: the verifier is only one of four boundaries, and the narrowest one; the study should be structured around the load gate, per-(arch, version, config) verifier behavior, BTF/CO-RE portability, performance cost, and the visibility gap. The same question surfacing in two independent communities suggests this is a genuinely recurring gap in the field's public literature, not a one-off ask.

### The kernel mailing list is actively patching the same boundary

The public kernel archive was active today on precisely this theme. A multi-patch series fixes a buffer overflow in the powerpc JIT for large BPF programs and explicitly targets missing verifier selftest coverage on that architecture — direct evidence that per-architecture verification and test debt is a live problem, and the kind of boundary a cross-architecture study should measure. An RFC proposes UHMUL and SHMUL multiply instructions for the instruction set, and a separate RFC proposes tracking scalar equality across the low 32 bits, continuing active work on verifier scalar range tracking. A large patchset extends BPF with by-value struct and `__int128` arguments, including for kfuncs, and a bpf-next series adds BPF-driven proactive memcg reclaim through a new kfunc. A sockmap fix addresses double-counting of self-redirect sequence numbers. The shared thread: the verifier and its ABI are evolving faster than documentation and downstream assumptions, which is exactly why the boundary matrix needs a version axis.

### The practitioner forum: performance and behavior reconstruction

The public forum's week was dominated by production engineering rather than security. A two-part public replatforming series — the second part detailing eleven gaps in the Linux ecosystem that a large operator closed with eBPF — [continued on the foundation blog](https://ebpf.io/blog/cloudflare-replatforming-2/). A process-behavior reconstruction tool correlated raw syscall events into a behavior timeline, and two threads covered squeezing performance out of eBPF and making BPF policies fast and space-efficient. A CNI author shared a ClusterIP host-routing data path whose first debugging step was reverse-path filtering — the same boundary explored in the [2026-09-03 answer](/ebpf-qa/2026-09-03-ebpf-kubernetes-rp-filter-return-traffic/). The recurring expectation across these threads: quantify instruction and data-path cost, and show that low-level events correlate into a reproducible explanation. That is also the operational lesson for the security boundary above: performance baselines and BPF-native visibility baselines are what make boundary 4 measurable in the first place.
