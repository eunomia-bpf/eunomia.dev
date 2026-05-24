# KubeCon + CloudNativeCon North America 2026 — Talk Proposals (eunomia-bpf)

> **Logistics:** KubeCon + CloudNativeCon **North America 2026 — Nov 9–12, Salt Lake City, UT.**
> **CFP deadline: May 31, 2026, 11:59pm Mountain Time (UTC-7).** ⚠️ ~1 week out as of 2026-05-24.
> Rule: do **not** appear as speaker/co-speaker on **more than 3 submissions** (all your
> submissions get pulled if you exceed it) — these two count toward that cap.
>
> Target venue: **KubeCon + CloudNativeCon North America 2026.** The landscape notes
> below are drawn from the most recent KubeCon/CNCF programming (incl. EU 2026), since the
> NA 2026 schedule isn't fully published yet — the thematic trends (agents/MCP, eBPF,
> runtime security) and the *gap* we exploit carry across both. Re-check the NA 2026 CFP
> tracks + deadline before submitting (see bottom of file).

Two submissions, deliberately non-overlapping. They share a worldview (the OS/eBPF
layer is the missing substrate for AI agents) but make two *different* contrarian
claims and target different tracks, so they do not cannibalize each other.

| | Proposal 1 | Proposal 2 |
|---|---|---|
| One-line claim | **Verify agent side effects below the framework.** | **Don't treat the pod as the agent's execution unit.** |
| Angle | Security boundary realignment | Execution substrate / "agent OS" |
| Center sentence | The framework sees what the agent *asked* to do; the environment must verify what *actually happened*. | Kubernetes sees a pod; the OS sees what the agent really is — a dynamic process tree with tool-call-level behavior. |
| Projects | AgentSight, ActPlane, eBPF evidence plane | Sandlock, AgentCgroup, ActPlane (light) |
| Recommended submission | Session Presentation, not poster | Poster Session |
| Track / topic | Security | AI Inference + Agentic |
| Level | Intermediate | Intermediate |
| Speakers | eunomia-bpf | eunomia-bpf + Cong Wang if confirmed |

---

## CFP-ready form copy

The CNCF CFP asks for concise, schedule-ready descriptions: focused problem, concrete
technical contribution, relevance to the ecosystem, no marketing pitch, and a session
description under 1000 characters. The copy below is written for Sessionize fields first;
the longer notes later in this file are for preparation and demo planning.

### Submission 1 - Session Presentation

**Session Title**

When the Agent Framework Is Not Yours: eBPF Runtime Evidence for AI Agents on Kubernetes

**Are you submitting for a Poster Session?**

No

**Session Format**

Session Presentation

**Track**

Security

**Level**

Intermediate

**Description - under 1000 characters**

AI agents are no longer simple scripts inside a pod. They are increasingly delivered as
complex third-party frameworks that own prompts, tool routing, approvals, MCP access,
memory, retries, logs, and sometimes sandbox policy. Meanwhile, the assets they act on -
repos, CI runners, clusters, secrets, and networks - belong to the platform
team. This ownership and trust split changes the security model: framework approvals,
logs, and provider-managed sandbox settings cannot be the environment owner's source of
truth.

This session uses a three-layer model: agent framework, execution environment/sandbox, and
environment-owned evidence/policy plane. With eBPF, Kubernetes metadata, and runtime
policy, we verify what happened below the framework: process lineage, commands,
file/secret access, network egress, credential use, and subprocess behavior. A demo
compares framework logs, sandbox controls, and OS evidence as prompt injection drives
credential access and unexpected egress.

**Benefits to the ecosystem**

Cloud-native teams need to run agents they do not fully own: coding assistants, MCP
clients, CI agents, and SaaS-managed runtimes with access to repos, clusters, secrets, and
networks. Current patterns often over-trust tool approvals, framework logs, or sandbox
settings supplied by the agent runtime.

This session gives platform and security engineers a reusable boundary model for deciding
what can be authorized at the tool layer, what can be isolated at the sandbox layer, and
what must be independently verified at the OS/runtime layer. It helps move agent operations
toward auditable runtime controls built on Kubernetes and eBPF without claiming that eBPF
replaces sandboxing or MCP authorization.

**Case study?**

No

**Presented this talk before?**

No. This proposal is new for KubeCon + CloudNativeCon North America 2026.

**CNCF-hosted software**

Kubernetes. Add Cilium, Tetragon, or OpenTelemetry only if the demo directly uses them.

**Open source projects**

eunomia-bpf, AgentSight, ActPlane.

**Additional resources**

- Project/demo repository: add public GitHub link before submitting.
- Prior speaking sample: add one KubeCon, Linux Plumbers, or OSS Summit recording.

### Submission 2 - Poster Session

**Session Title**

Beyond Pods: OS-Level Runtime Primitives for Fast, Stateful, and Adaptive AI Agents

**Are you submitting for a Poster Session?**

Yes

**Poster Session Topic**

AI Inference + Agentic

**Level**

Intermediate

**Description - under 1000 characters**

AI agents on Kubernetes do not behave like microservices. An agent is a control loop that
repeatedly calls tools, forks subprocesses, loads runtime
state, retries failed actions, and changes strategy from feedback. Kubernetes sees a pod;
the operating system sees a dynamic process tree with tool-call-level behavior.

This poster proposes an OS runtime substrate for agentic workloads and connects three
open-source prototypes: Sandlock for process-level copy-on-write sandboxing and fast state
cloning, AgentCgroup for tool-call-aligned resource control and eBPF enforcement, and
ActPlane for execution contracts that relate agent actions to files, subprocesses, and
network effects. It shows the architecture, measurements for forkable sandbox startup and
memory-spike behavior, and Kubernetes-facing primitives including ToolCallScope,
ResourceHint, CorrectiveFeedback, and ExecutionContract.

The goal is to make agents faster, more efficient, and easier to operate at scale.

**Benefits to the ecosystem**

Agentic workloads are arriving faster than the platform abstractions around them. Treating
each agent as only a pod hides the tool-call boundaries, subprocess trees, warm state, and
bursty memory behavior that determine cost and reliability. This poster gives the cloud
native ecosystem an early, concrete vocabulary for these runtime needs and invites feedback
before APIs or operational patterns harden.

The work is relevant to platform engineers building internal agent platforms, researchers
studying agent runtime behavior, and Kubernetes contributors thinking about future workload
primitives. By presenting measurements and prototypes instead of a product pitch, the
poster can help turn scattered agent runtime problems into shared, testable infrastructure
questions.

**Case study?**

No

**Presented this talk before?**

No. This poster proposal is new for KubeCon + CloudNativeCon North America 2026.

**CNCF-hosted software**

Kubernetes.

**Open source projects**

Sandlock, AgentCgroup, ActPlane, eunomia-bpf.

**Additional resources**

- Sandlock paper/blog/repository: add public link before submitting.
- AgentCgroup paper/repository: add public link before submitting.
- ActPlane repository: add public link before submitting.
- Prior speaking sample: add one KubeCon, Linux Plumbers, or OSS Summit recording.

---

## Context: what recent KubeCon/CNCF is actually programming (trend basis for NA 2026)

- **Agents are a headline theme, not a novelty.** CNCF ran an **Agentics Day (MCP + Agents)**
  focused on real-world implementations, production deployment, governance, and how
  platform teams operate agentic capabilities. AI, Security, Platform Engineering, and
  Emerging + Advanced are all called out as priority directions.
- **Agent security today lives at the control plane / tool layer**: agent identity &
  delegation (SPIRE, Keycloak, OAuth2, MCP Gateway — e.g. IBM Research's "When an Agent
  Acts on Your Behalf, Who Holds the Keys?"), MCP/tool authorization (agentgateway +
  Kyverno "Least-Privilege for AI"), and agentic K8s ops/chaos.
- **eBPF talks are abundant but classic**: Cilium positioning as the AI-workload data
  plane, Tetragon runtime security, DNS tracing via eBPF in OTel OBI, eBPF attack-flow
  reveal, GPU observability with eBPF.
- **The gap (our opening):** nobody has landed a mainstage talk that treats **AI agent +
  OS/eBPF boundary** as the core architecture. Microsoft's Agent Governance Toolkit even
  uses an OS analogy ("think of it as the kernel for AI agents") but its README admits
  enforcement is a **Python middleware layer, not OS-kernel level**, and recommends
  per-agent containers for OS-level isolation. That gap is exactly what these two talks fill.

> Positioning rule for both CFPs: **do not** write "AI + eBPF is cool." Write the missing
> layer. P1 = "agent security stops at MCP/IAM — here is the missing runtime evidence
> layer." P2 = "agent operations stop at the K8s API/pod — here is the missing OS
> execution substrate."

---

## Proposal 1 — Security

### Title
**When the Agent Framework Is Not Yours: eBPF Runtime Evidence for AI Agents on Kubernetes**

(Alternatives — A: *Securing Third-Party AI Agents in Your Cluster with eBPF Runtime
Evidence*; B: *Agent Frameworks Are Not Trust Boundaries: eBPF Runtime Security for
Kubernetes*. The title above is the most accurate for the ownership-boundary argument.)

### One-line pitch
As AI agents move into Kubernetes, platform teams cannot assume the agent framework,
harness, or sandbox is trustworthy or modifiable. This talk shows how to build an
**environment-owned evidence plane** using eBPF to verify agent behavior below that stack.

### Core claim
MCP gateways and framework permissions **authorize intended actions**; sandboxing
**isolates execution**; eBPF/runtime evidence **verifies actual side effects**. The
agent framework/harness is part of the workload, **not automatically part of the trusted
computing base (TCB)**.

### Why it's novel (not "another agent sandbox / MCP gateway / eBPF observability demo")
**Security boundary realignment for third-party agents:**
- The **agent harness** (prompt loop, tool routing, approvals, retries, memory, logs)
  belongs to the **agent provider**.
- The **execution environment** (repo, cluster, CI runner, secrets, network) belongs to
  the **platform owner**.
- Therefore **security evidence must belong to the environment owner**, and the trusted
  layer must sit **below or outside the harness**.

This reframes the boundary the way today's reality forces it: developers *bring* Claude
Code / Cursor / Codex CLI / OpenHands / LangGraph / MCP clients into your cluster. You
control pods, network policy, secrets, runtime security — but you **cannot** modify the
third-party harness's prompt loop / approval classifier / tool router / log format, and you
**cannot assume** its self-reported action log is complete. (Even Anthropic's own docs say
they don't security-audit every MCP server and recommend external VM isolation for
untrusted content — the official model is *not* "trust the harness and done.")

### Threat model (three adversaries)
1. **Compromised agent** — prompt injection via malicious repo/issue/webpage/MCP response
   drives unintended commands.
2. **Untrusted / opaque harness** — not malicious, but the environment owner can't verify
   its internal state (permission classifier, approvals, retries, MCP routing, hooks).
   Harness hooks deny/ask/allow *inside* the harness pipeline — not as an external,
   platform-enforced boundary.
3. **Sandbox escape / policy gap** — container breakout, network bypass, credential leak,
   subprocess side effects. (AISI's *SandboxEscapeBench* uses "sandbox-within-a-sandbox"
   — single-layer sandbox is no longer a sufficient assumption.)

eBPF/OS value here is **not** "replace the sandbox." It is the **environment-owned,
harness-external observation + enforcement layer**:
> Sandbox constrains *where* actions can happen. eBPF records and enforces *what* actually happens.

### Talk structure (do NOT open with eBPF — open with the boundary mismatch)
1. **The new split: agent provider vs environment owner.** An agent isn't an ordinary
   container app; it's a semi-autonomous control loop carrying its own harness.
2. **Harness is not a security boundary** — third-party/SaaS-managed, self-reported logs,
   policy-semantics mismatch, too-high-level action abstraction, injection-influenced.
   *Center sentence of the talk.*
3. **Sandbox is necessary but not enough** — isolation ≠ complete evidence; policy gaps
   (mounts, egress, credential proxy, child processes, breakout).
4. **Evidence plane design** (the architecture diagram below) — independent of harness
   cooperation: process tree, file/secret access, network egress, credential/metadata
   access, tool side-effect correlation, harness-log vs OS-evidence mismatch.
5. **Demo.**

```
Third-party Agent Harness        (owned by agent provider — UNTRUSTED tenant)
  prompt loop / tool routing / MCP client / self-reported logs
        |
        v
Execution Environment            (owned by platform team)
  container / VM / namespace · repo / CI / cluster creds · MCP servers / shell / network
        |
        v
Independent Evidence Plane       (owned by environment owner — the TCB)
  eBPF process lineage · file/network/syscall events · K8s metadata
  policy engine · audit log / incident replay
```

### Demo (very KubeCon)
Platform team lets a third-party coding agent fix a bug. Agent runs in a K8s job/pod, repo
mounted, allowed to reach GitHub + package registry + a test service. A prompt-injection
payload hidden in an issue/comment drives the agent to read `.env` / the service-account
token and `curl` it out. Compare three defenses:
- **MCP/tool log**: only sees "agent called bash / a file tool."
- **Sandbox**: may block some paths, but depends on configuration.
- **eBPF evidence plane**: sees process lineage, secret read, network egress, K8s identity
  → flags the policy violation and the intent/effect mismatch.

### Hard line for the CFP
> Current agent security often assumes the harness can enforce policy. That assumption
> breaks when platform teams run third-party agents — coding assistants, MCP clients,
> SaaS-managed runtimes. In those cases **the harness is part of the workload, not part of
> the trusted computing base.**

### Nuance to keep correct
Not "you can never modify the sandbox" — rather **"you can't assume the sandbox belongs to
the environment owner."** If Claude Code runs in *your* machine/pod you can wrap it in
container/VM/eBPF/network policy; but for **provider-managed execution** (e.g. Claude Code
on the web runs in an Anthropic-managed isolated VM with scoped credentials/proxy), the
environment owner only controls the interface, repo permissions, network/proxy, and audit
logs — you can't drop your own eBPF agent inside that VM. The talk should state both cases.

---

## Proposal 2 — Execution substrate ("Agent OS", non-security)

### Title
**Beyond Pods: OS-Level Runtime Primitives for Fast, Stateful, and Adaptive AI Agents**

(Alternative, more paper-ish: *Agents Are Process Trees, Not Pods: Building an OS Runtime
Substrate for AI Agents on Kubernetes*. The "Beyond Pods" title reads more KubeCon.)

### Strong thesis (architecture, not a feature list)
> Cloud-native AI agents need an **OS runtime substrate**, not just Kubernetes APIs and
> sandboxes. The pod is too coarse and the tool call is too invisible. To run agents
> efficiently, the system must manage agent execution at the level of **process trees,
> tool-call boundaries, copy-on-write state, and adaptive resource contracts.**

Security is only a *side benefit* here; the main line is **fast / stateful / adaptive /
efficient**. (Keep ActPlane/contracts light so this doesn't slide back into a security talk.)

### Why the thesis holds — three mismatches
1. **Lifecycle mismatch.** K8s sees a pod lifecycle; the agent's real lifecycle is
   `reason → tool call → subprocess tree → observe result → retry/branch`.
   **Sandlock:** agents are process trees at the OS layer; wrapping every execution in a
   container/microVM re-initializes the environment, while process-level sandboxing inherits
   warmed state via `fork`/COW (model weights, dataset, interpreter state, JIT cache shared
   copy-on-write). Demonstrated **1000 sandbox forks in 718 ms.** This is a *runtime
   performance* model, not just security.
2. **Resource mismatch.** **AgentCgroup:** OS-level execution (tool calls + container/agent
   init) is **56–74% of end-to-end task latency**; **memory, not CPU**, is the multi-tenant
   concurrency bottleneck; tool-call-driven memory spikes reach a **15.4× peak-to-average
   ratio.** Pod-level requests/limits are too coarse and HPA/VPA too slow → align cgroup
   structure with tool-call boundaries; in-kernel enforcement via `sched_ext` and
   `memcg_bpf_ops`; agent uses resource hints / corrective feedback to adapt.
3. **Contract mismatch.** Traditional OS/K8s manage processes, containers, service accounts,
   network policy. Agents need workflow-level semantics: what workflow is allowed, which
   derived data may flow where, how to correct after a failed tool call. **ActPlane:**
   contracts authored at the harness level but **enforced at the OS/eBPF layer** over
   exec/file/network/syscall — information-flow/provenance contracts, a *deterministic
   operating contract for cooperative-but-forgetful agents* (not a static allowlist).
   *Mention lightly — don't let it dominate.*

### Talk outline
1. **Kubernetes sees pods; agents execute as process trees** — why agent workloads differ
   from microservice / serverless / batch.
2. **Cold-start and state are now OS problems** — Sandlock: fork/COW, warm state, branchable
   execution, *sandbox as a function call* (not an infrastructure service).
3. **Resource control must follow tool-call boundaries** — AgentCgroup: 56–74% OS execution
   time, 15.4× memory spike, tool-call cgroups, `sched_ext` / `memcg_bpf_ops`, agent
   feedback loop.
4. **Agent runtime needs contracts, not just limits** — ActPlane (kept light): OS observes
   derived processes/files/network effects against workflow/provenance contracts.
5. **What Kubernetes should expose next** (highest KubeCon value — a *mental model*, not a
   standardization push):
   - `AgentExecution` — an agent task/session ≠ a pod.
   - `ToolCallScope` — resource/permission/lifecycle unit of a tool call.
   - `ForkableSandbox` — COW-clonable warmed execution state.
   - `ResourceHint` — agent/harness declares intent (memory-high / network-none / compile-heavy).
   - `CorrectiveFeedback` — OS → agent signal ("test runner hit `memory.high`; retry with subset").
   - `ExecutionContract` — provenance/workflow contract (derived files can't leave workspace).

### The distinctive idea
> The operating system should become the **execution substrate for agentic workloads**:
> fork state, scope tool calls, enforce resources, and **feed runtime signals back into the
> reasoning loop.**

Unlike traditional workloads, an agent can *consume OS feedback* — declare resource needs
and reconstruct execution strategy from system signals (AgentCgroup's corrective-feedback
point). That's what separates an "agent OS" from ordinary OS resource control.

### Co-presenter division (with Cong Wang)
- **Cong Wang / Sandlock:** process-level sandbox + COW fork + fast state cloning.
- **eunomia-bpf:** AgentCgroup (eBPF resource control) + ActPlane (execution-contract plane, light).
- **Shared thesis:** agent runtime should be **OS-native, not container-only.**

---

## Reviewer-facing risks (address these in the CFPs / dry runs)
1. **KubeCon EU is demo-heavy and competitive.** Both talks need a *working live demo*, or
   they read as paper talks and get cut. P1's three-way defense comparison and P2's
   fork-1000 / memory-spike demos are the differentiators — build them first.
2. **P2 can look like "three research papers stapled together."** Frame Sandlock /
   AgentCgroup / ActPlane as **operator-facing primitives**, not project tours. The "What
   K8s should expose next" section is what makes it a platform talk, not a paper.
3. **The numbers (56–74%, 15.4×, 718ms/1000 forks) are the strongest ammo — reviewers will
   ask how to reproduce.** Name the setup/benchmark briefly in the CFP so they read as
   measured, not marketing.
4. **Co-speaker credibility helps acceptance** — state the Cong Wang collaboration and the
   clean division of labor explicitly in P2's CFP.
5. **Avoid cannibalization in submission metadata:** P1 → Security / Runtime Security /
   Observability; P2 → AI + ML / Platform Engineering / Emerging + Advanced. Same worldview,
   different nets — don't weave the same mesh twice.

## Sources / claims to verify before submitting
- Project repos: AgentSight, ActPlane (`github.com/eunomia-bpf/ActPlane`), Sandlock,
  AgentCgroup — confirm public status, exact metrics, and that each README backs the claim
  cited above.
- AgentCgroup paper: confirm the 56–74% / 15.4× / memory-bottleneck numbers and the
  `sched_ext` + `memcg_bpf_ops` enforcement details.
- Sandlock blog/paper: confirm the 1000-fork / 718 ms figure and the COW-shared-state claim.
- Anthropic Claude Code docs: permission model, sandboxed bash tool (OS-level FS/network
  isolation incl. spawned subprocesses), MCP non-audit statement, web = Anthropic-managed VM.
- KubeCon EU 2026 / CNCF: Agentics Day framing, track list; CiliumCon AI-data-plane +
  Tetragon positioning; Microsoft Agent Governance Toolkit "kernel for AI agents" + the
  Python-middleware-not-kernel caveat; AISI SandboxEscapeBench.
