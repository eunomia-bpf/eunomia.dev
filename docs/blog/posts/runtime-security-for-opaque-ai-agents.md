---
date: 2026-05-25
slug: runtime-security-for-ai-agents
---

# Runtime Observability and Enforcement for Opaque AI Agents: Beyond Sandboxes and Approvals

AI coding agents now run for hours, complete entire features end-to-end,
optimize production GPU kernels, and merge thousands of pull requests
autonomously. Meanwhile, most agent security still relies on human-in-the-loop
approval, and Anthropic's own data shows users approve 93% of prompts without
meaningful review. The result is predictable: products add bypass modes, users
disable permission gates, and 65% of firms report agent security incidents.

But the deeper problem is not approval fatigue. It is that the agent harness
(the prompt loop, tool routing, permission logic, and sandbox defaults) is
increasingly a third-party product the platform team did not write, running in a
sandbox the platform team may not own. The harness is not a trusted security
boundary. This post argues for separating agent security into three layers with
three different owners: intent authorization (harness-owned), execution
isolation (ownership contested), and side-effect verification (must be
platform-owned). When the layers agree, you have confidence. When they
disagree, you need independent observability and enforcement at the OS level to
detect it, and that is exactly the layer most agent platforms are missing. We
are exploring this direction with two open-source projects:
[AgentSight][agentsight] for runtime observation and
[ActPlane][actplane] for runtime enforcement and harness, both using eBPF to provide an
independent evidence and policy plane below the agent harness.

<!-- more -->

## Why Now: Complexity Up, Guardrails Behind

The important change in 2026 is not that agents exist. It is the scale and
duration of what they do.

A year ago, the typical agent task was "fix this bug" or "write this function."
In 2026, agents routinely run for hours on complex, multi-step work. OpenAI
documented a Codex session that [ran for 25 hours uninterrupted][codex-long],
consuming 13 million tokens and producing 30,000 lines of code from a blank
repository. Anthropic's agentic coding report cites a [12.5-million-line
codebase change completed in a single 7-hour run][anthropic-trends]. Meta's
[KernelEvolve][kernelevolve] uses multi-agent coordination to write and optimize
production GPU kernels, compressing work that previously required weeks of
expert systems engineering into hours. On SWE-bench Verified, [top agents now
resolve 60–70%][swebench] of real GitHub issues, up from under 30% in early
2024. Devin has [merged hundreds of thousands of pull requests][devin-review]
across enterprise customers with a 67% merge rate. Goldman Sachs [deployed
hundreds of Devin instances][goldman] across a 12,000-person engineering team.

These are not research demos. They are production workflows: background tasks,
parallel execution, multi-hour sessions, end-to-end feature development, kernel
optimization, and enterprise-scale code changes.

**Meanwhile, the guardrails designed to keep agents safe have not kept pace.**

Most agent security still relies on human-in-the-loop approval: a prompt asks
the user to approve or deny each action before it executes. This works for short
sessions with a few tool calls. It does not work when an agent makes hundreds of
decisions over hours of autonomous operation.

The evidence suggests that approval-based control is already failing in
practice. Anthropic's own data shows that [Claude Code users approve 93% of
permission prompts][anthropic-auto], a rate consistent with rubber-stamping
rather than meaningful review. An independent stress test of Claude Code's auto
mode found an [81% false negative rate][permission-gate] on ambiguous
state-changing actions, meaning the classifier allowed 4 out of 5 actions that
should have required human review. Real incidents have followed: in documented
cases, users running agents without permission gates had their [home directories
deleted][yolo-incidents] by `rm -rf` commands the agent generated. A 2026
industry survey found that [65% of firms reported AI agent security
incidents][kiteworks], with most involving organizations lacking proper agent
access controls.

Products have responded by adding bypass mechanisms. Claude Code offers
`--dangerously-skip-permissions`. Windsurf's Cascade agent [proceeds
autonomously][windsurf-cursor] where Cursor stops to ask. Community guides now
focus on "how to safely use YOLO mode." Anthropic researcher Nicholas Carlini
ran [16 parallel Claude agents with permissions bypassed][carlini], with the
caveat: "Run this in a container, not your actual machine."

This is the tension: **the more capable agents become, the more users want to
let them run uninterrupted, and the less effective human-in-the-loop becomes as
the primary security boundary.**

That tension is what creates the need for a different security model.

## The Accountability Gap

The deeper issue is not just that agents are more capable. It is that the agent
harness, the component that decides what the agent does, is increasingly a
third-party product the platform team did not write.

A modern agent harness is not a thin wrapper around a model. It includes a
prompt loop, planning and retry logic, tool routing, MCP clients, permission
modes, approval gates, hooks, memory, logs, credential handling, and sometimes
sandbox defaults. In many deployments, that harness comes from a hosted
coding-agent service or an open-source framework the platform team does not
control.

This is already visible across the ecosystem. GitHub Copilot's [coding
agent][copilot-agent] runs autonomously in GitHub Actions, researching
repositories, creating plans, making changes, and opening pull requests. OpenAI
[Codex][codex] runs background tasks in sandboxed cloud environments with
controlled network access. Claude Code runs cloud sessions in Anthropic-managed
VMs with scoped credentials. Kubernetes SIG is defining [Agent
Sandbox][k8s-sandbox] for isolated, stateful agent workloads. Recent research
datasets show [agent-authored pull requests at scale][aidev] across real
repositories.

The ownership split is now explicit in major platforms. Anthropic's shared
responsibility framework [divides agent security into four
layers][anthropic-shared-resp] (Model, Harness, Tools, Environment) and
states that the deploying organization owns three of the four. GitHub's agentic
workflow architecture starts from the premise that ["agents cannot be trusted by
default, especially in the presence of untrusted inputs"][github-agentic-sec],
using kernel-enforced communication boundaries that hold even if the agent
container is compromised. OpenAI's Codex documentation [acknowledges][codex-security]
that "devcontainers provide substantial protection, but they do not prevent
every attack."

The platform team still owns the repository, the CI runner, the Kubernetes
cluster, the service accounts, the secrets, and the internal network. But the
runtime acting on those assets may be opaque.

There is also a second split that matters even more for platform teams: **the
sandbox may not be controlled by the environment owner either.** If the agent
runs in a provider-managed cloud (Claude Code on the web runs in
[Anthropic-managed isolated VMs][claude-security] with scoped credential
proxies; Codex runs in [OpenAI-managed containers][codex-sandbox]), the
platform team cannot attach its own monitoring, modify isolation policy, or
inspect the sandbox internals. Even Anthropic's own managed agent architecture
explicitly [decouples the "brain" (Claude + harness) from the
"hands" (sandboxes)][anthropic-managed], treating containers as disposable
cattle and ensuring tokens are never reachable from the sandbox where generated
code runs. This is good architecture, but it is the provider's architecture,
not the platform team's.

When agents run locally or on self-hosted infrastructure (GitHub now [supports
self-hosted runners][copilot-self-hosted] for its coding agent, and Kubernetes
Agent Sandbox provides [gVisor/Kata-backed isolation][k8s-sandbox] under the
platform operator's control), the environment owner can wrap the agent in its
own sandbox and evidence layer. When agents run in provider-managed
environments, the independent evidence plane must move to the boundaries the
platform team does control.

This creates the accountability gap: **the platform team is responsible for
production impact from a workload it cannot fully inspect, running in a sandbox
it may not own.**

The old mental model was simple: the agent is risky, so put it in a sandbox.
The new reality has a different trust boundary: the agent and its harness are
part of the workload, and the environment owner needs an independent evidence
plane.

## Three Layers, Three Questions

MCP, sandboxes, and OS-level evidence are all necessary for agent security.
They are not interchangeable. Each answers a fundamentally different question,
and each has a different owner.

**Intent authorization** (MCP, tool gateways, approval prompts) answers: what
is the agent *supposed* to do? Which tools may it call, under which identity,
with which scopes? This is the right place to enforce access control before a
dangerous action happens. But a tool approval is not proof of side effects. A
framework log saying "run tests" does not prove that the process tree only ran
tests. An MCP server can be well-authenticated and still be part of a workflow
that causes unexpected local effects. This layer is typically owned or mediated
by the agent harness.

**Execution isolation** (containers, VMs, network policy, namespaces) answers:
what *can* the agent reach? Which files, network endpoints, credentials, and
syscalls are available? This is the right place to limit blast radius. But a
sandbox does not automatically record what the agent attempted within its
constraints: which process read a secret, which subprocess opened a network
connection, whether the sandbox policy matched the approved intent. This layer's
ownership is contested: it may belong to the agent provider, the platform team,
or both.

**Side-effect verification** (OS/runtime evidence) answers: what *actually
happened*? Which processes ran, which files were read, which network connections
were opened, which credentials were accessed? This layer provides facts about
execution, independent of what the framework reported or the sandbox intended.
This layer must be owned by the environment operator. Otherwise there is no
independent source of truth.

The security model is the combination:

```text
authorize intent  →  isolate execution  →  verify side effects
(harness-owned)      (ownership contested)  (must be platform-owned)
```

When all three layers agree, you have confidence. When they disagree, you need
OS-level observability and controls, independent of the harness, to detect the
mismatch, contain the damage, and reconstruct what happened.

## Why Independence Matters

The reason to keep these layers independent follows from the trends above, but
also from a deeper structural argument about ownership and trust.

### Approval fatigue

As we have seen, users approve 93% of prompts and products actively offer bypass
modes. When approvals are relaxed, the other two layers must compensate.
If you auto-approve routine actions, you need an independent way to verify what
those actions actually did. If you bypass permissions for speed, you need
stronger containment and stronger evidence.

### Harness opacity

When the harness is opaque, application-level telemetry cannot be the sole
evidence source. OpenTelemetry GenAI conventions and framework-level tracing are
valuable when you own the framework. But opaque agent apps, closed-source
runtimes, hosted execution, stripped binaries, and arbitrary subprocess trees
can all break the assumption that the framework trace is complete. Security
researchers have already found [30+ vulnerabilities across all major AI
IDEs][idesaster] (Cursor, Copilot, Windsurf, Claude Code), enabling data theft
and remote code execution through prompt injection into agent tool chains.

The MCP layer records intended tool calls. The OS layer records actual side
effects. When the harness is opaque, the gap between these two is exactly where
security incidents live.

### The trust boundary is an ownership boundary

The deepest reason for independence is that the three layers serve different
owners with different incentives.

The harness provider's goal is to complete the user's task: maximize
autonomous coding productivity, reduce permission friction, deliver results.
The platform team's goal is to protect the repository, secrets, cluster,
CI runner, internal network, and production APIs. These goals are not opposed,
but they are not identical. When they conflict, when the fastest path to task
completion involves reading credentials, opening network connections, or
modifying files outside the workspace, the harness will optimize for
completion unless an independent boundary stops it.

This is why [Bhattarai and Vu argue][deterministic-boundaries] that
"probabilistic compliance is not compliance": training-based and
classifier-based defenses may reduce empirical attack rates, but cannot provide
deterministic guarantees under adversarial conditions. Only architectural
enforcement can. Red Hat's experience deploying multi-agent systems on Kagenti
frames the same insight differently: this is ["a multi-tenancy problem disguised
as an AI problem"][redhat-kagenti]. The agent is an untrusted tenant. The
platform needs the same kind of isolation, identity, and audit controls it would
apply to any untrusted workload.

The [OWASP Top 10 for Agentic Applications][owasp-agentic] reinforces this
framing. Its top risk (ASI01, Agent Goal Hijacking) is that "agents cannot
reliably distinguish instructions from data," and a single malicious input from a
repository, issue, MCP response, or web page can redirect the agent to perform
harmful actions using its legitimate tools. This is not a hypothetical:
[Bishop Fox demonstrated][bishopfox-deputy] confused deputy attacks where
instructions embedded in support tickets caused agents to exfiltrate data using
authorized tools, with "the user's name on every audit log entry." [Docker
documented][docker-mcp-horror] a GitHub prompt injection chain where a
malicious issue hijacked an MCP-connected agent to steal confidential data from
private repositories.

The threat model for platform teams therefore has three adversary categories:

| Threat | Which layer fails | Evidence plane detects |
|--------|------------------|-----------------------|
| **Compromised agent** (prompt injection, malicious repo/issue/MCP response) | Intent layer: agent is tricked into unintended actions | Actual side effects diverge from stated intent |
| **Untrusted harness** (opaque permission logic, incomplete logs, unauditable internal state) | Cannot verify harness completeness | OS-level facts independent of harness reporting |
| **Sandbox escape or policy gap** (container breakout, mounted credentials, network bypass) | Isolation layer fails or is misconfigured | Detects behavior outside expected sandbox boundary |

AISI's [SandboxEscapeBench][sandbox-escape] makes the third category concrete:
frontier models can reliably escape container sandboxes under
misconfigurations that plausibly occur in real systems, and the researchers
discovered four unintended escape paths the benchmark designers had missed.
Their recommendation: ["treat plain Docker isolation as insufficient by
default."][sandbox-escape]

In all three cases, the OS/runtime evidence layer is the independent control
that lets the platform team detect the problem, regardless of which other layer
failed.

## What OS Evidence Looks Like

At the OS/runtime layer, evidence includes:

- **Process lineage**: the full tree from agent to subprocess to network call
- **File access**: which paths were read or written, including credential paths
- **Network behavior**: connections, destinations, timing, data volume
- **Container metadata**: namespace, cgroup, pod identity, service account
- **Subprocess behavior**: commands that bypass framework instrumentation

This evidence is collected below the application layer, typically via eBPF,
audit subsystems, or kernel instrumentation. It does not require modifying the
agent app. Its key property is independence: the evidence is owned and collected
by the environment operator, not by the agent provider.

This makes cross-layer comparison possible:

```text
Framework report:    run tests
Sandbox policy:      workspace mounted, registry allowed, SA token mounted
OS evidence:         agent → shell → python → curl
                     read: /var/run/secrets/.../token
                     connect: unknown external host
```

Each layer saw a different part of the event. The OS evidence layer is what lets
the platform team detect the mismatch.

## Deployment Reality

OS-level evidence is strongest when you control the host, node, or VM where the
agent executes. If the agent runs entirely in a provider-managed environment,
you may not be able to attach eBPF inside it.

In that case, the same model applies, but evidence shifts to the boundaries you
do control:

- Repository permissions and branch protection
- Scoped credentials with minimal lifetime
- CI/CD and GitHub audit logs
- Network proxies and webhook events
- Artifact access logs
- Provider-supplied session logs

This evidence is weaker than owning the runtime boundary, but it is still better
than treating the agent transcript as the only source of truth.

The design question for platform teams is:

> Where is the lowest layer I actually control?
> That is where the independent evidence plane should live.

## Practical Checklist

If you are building or evaluating an agent platform, ask these questions at
each layer.

**Intent authorization (MCP / tool access):**

- Are MCP servers allowlisted?
- Are OAuth scopes minimal and audience-bound?
- Are local MCP servers treated as code execution risk?
- Are high-risk tools gated by human approval?
- Are tool calls logged with enough context for audit?

**Execution isolation (sandboxing):**

- Is filesystem access default-deny or broad workspace mount?
- Can the agent reach cloud metadata endpoints?
- Is network egress restricted by domain, IP, or proxy?
- Are service account tokens mounted into the environment?
- Are process, memory, CPU, and runtime duration bounded?
- Who owns the sandbox policy: the platform team or the agent provider?

**Side-effect verification (runtime evidence):**

- Can you reconstruct process lineage for an agent session?
- Can you see file and credential access below the framework?
- Can you correlate network egress with pod, service account, and command?
- Can you detect mismatch between tool intent and OS side effects?
- Can you replay an incident without trusting only framework logs?

**Guardrail integration:**

- Which side effects should be blocked immediately?
- Which should trigger alert or human review?
- Which policies belong in MCP config, sandbox config, Kubernetes policy,
  eBPF/LSM, or network controls?
- What happens when framework logs and OS evidence disagree?

## Closing

Agent runtimes are becoming more capable, more managed, and more opaque. The
security model cannot depend on any single layer, especially when the layers
have different owners.

The harness is not a trusted boundary. The sandbox ownership depends on the
deployment model. The only layer the environment operator can guarantee it
owns is the OS/runtime evidence plane.

MCP authorizes intent. Sandboxes constrain execution. OS evidence verifies side
effects. Each is necessary; none is sufficient. The practical model is their
separation:

```text
authorize intent  →  isolate execution  →  verify side effects
(harness-owned)      (ownership contested)  (must be platform-owned)
```

The implementation details vary by deployment, but the separation, and the
ownership question, is the part that should remain stable.

We are building two open-source tools along these lines:
[AgentSight][agentsight] for eBPF-based runtime observation and
[ActPlane][actplane] for eBPF-based runtime enforcement of agent behavior.
They are possible implementations of the verification layer described here,
not the only ones.

## References

[codex-long]: https://developers.openai.com/blog/run-long-horizon-tasks-with-codex "Run long horizon tasks with Codex"
[anthropic-trends]: https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf "Anthropic 2026 Agentic Coding Trends Report"
[kernelevolve]: https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/ "KernelEvolve: Meta's Ranking Engineer Agent"
[swebench]: https://www.vals.ai/benchmarks/swebench "SWE-Bench Verified Leaderboard"
[devin-review]: https://cognition.ai/blog/devin-annual-performance-review-2025 "Devin's 2025 Performance Review"
[goldman]: https://www.cnbc.com/2025/07/11/goldman-sachs-autonomous-coder-pilot-marks-major-ai-milestone.html "Goldman Sachs autonomous coder pilot"
[anthropic-auto]: https://www.anthropic.com/engineering/claude-code-auto-mode "Claude Code auto mode: a safer way to skip permissions"
[permission-gate]: https://arxiv.org/abs/2604.04978 "Measuring the Permission Gate: Claude Code Auto Mode"
[yolo-incidents]: https://gist.github.com/hartphoenix/698eb8ef8b08ad2ce6a99cf7346cd7cc "Claude Code YOLO Mode incidents"
[kiteworks]: https://www.kiteworks.com/cybersecurity-risk-management/ai-agent-security-incidents-2026/ "AI Agent Security Incidents Hit 65% of Firms"
[windsurf-cursor]: https://stackbuilt.co/blog/windsurf-vs-cursor-2026 "Windsurf vs Cursor: Agent Autonomy vs IDE Precision"
[carlini]: https://x.com/nicholas_carlini "Nicholas Carlini on parallel agents"
[idesaster]: https://thehackernews.com/2025/12/researchers-uncover-30-flaws-in-ai.html "30+ Flaws in AI Coding Tools"
[copilot-agent]: https://docs.github.com/en/copilot/concepts/about-copilot-coding-agent "About Copilot coding agent"
[codex]: https://developers.openai.com/codex/cloud "OpenAI Codex cloud"
[k8s-sandbox]: https://agent-sandbox.sigs.k8s.io/ "Kubernetes SIGs Agent Sandbox"
[aidev]: https://arxiv.org/abs/2602.09185 "AIDev: Studying AI Coding Agents on GitHub"
[bessemer]: https://www.bvp.com/atlas/securing-ai-agents-the-defining-cybersecurity-challenge-of-2026 "Securing AI agents: the defining cybersecurity challenge of 2026"
[anthropic-shared-resp]: https://www.anthropic.com/research/trustworthy-agents-in-practice "Trustworthy Agents in Practice"
[github-agentic-sec]: https://github.blog/ai-and-ml/generative-ai/under-the-hood-security-architecture-of-github-agentic-workflows/ "Security Architecture of GitHub Agentic Workflows"
[codex-security]: https://developers.openai.com/codex/agent-approvals-security "Codex Agent Approvals & Security"
[codex-sandbox]: https://developers.openai.com/codex/concepts/sandboxing "Codex Sandbox Concepts"
[claude-security]: https://docs.anthropic.com/en/docs/claude-code/security "Claude Code Security"
[anthropic-managed]: https://www.anthropic.com/engineering/managed-agents "Scaling Managed Agents: Decoupling the Brain from the Hands"
[copilot-self-hosted]: https://github.blog/changelog/2025-10-28-copilot-coding-agent-now-supports-self-hosted-runners/ "Copilot coding agent supports self-hosted runners"
[anthropic-sandboxing]: https://www.anthropic.com/engineering/claude-code-sandboxing "Making Claude Code More Secure and Autonomous"
[deterministic-boundaries]: https://arxiv.org/abs/2602.09947 "Trustworthy Agentic AI Requires Deterministic Architectural Boundaries"
[redhat-kagenti]: https://next.redhat.com/2026/03/05/zero-trust-ai-agents-on-kubernetes-what-i-learned-deploying-multi-agent-systems-on-kagenti/ "Zero trust AI agents on Kubernetes with Kagenti"
[owasp-agentic]: https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/ "OWASP Top 10 for Agentic Applications 2026"
[bishopfox-deputy]: https://bishopfox.com/blog/otto-support-confused-deputy "The Confused Deputy: MCP Attack"
[docker-mcp-horror]: https://www.docker.com/blog/mcp-horror-stories-github-prompt-injection/ "MCP Horror Stories: GitHub Prompt Injection Data Heist"
[sandbox-escape]: https://arxiv.org/abs/2603.02277 "SandboxEscapeBench: Can AI Agents Escape Their Sandboxes?"
[safeharness]: https://arxiv.org/abs/2604.13630 "SafeHarness: Lifecycle-Integrated Security Architecture for LLM-based Agents"
[anthropic-nist]: https://www-cdn.anthropic.com/43ec7e770925deabc3f0bc1dbf0133769fd03812.pdf "Anthropic NIST RFI on Agentic Security"
[infoq-k8s-agents]: https://www.infoq.com/articles/securing-autonomous-ai-agents-kubernetes/ "Securing Autonomous AI Agents on Kubernetes"
[agentsight]: https://github.com/eunomia-bpf/agentsight/ "AgentSight repository"
[actplane]: https://github.com/eunomia-bpf/ActPlane "ActPlane repository"

- [GitHub Docs: About Copilot coding agent][copilot-agent]
- [GitHub: Security Architecture of Agentic Workflows][github-agentic-sec]
- [GitHub: Copilot coding agent supports self-hosted runners][copilot-self-hosted]
- [OpenAI Codex cloud documentation][codex]
- [OpenAI: Run long horizon tasks with Codex][codex-long]
- [OpenAI: Codex Agent Approvals & Security][codex-security]
- [Anthropic 2026 Agentic Coding Trends Report][anthropic-trends]
- [Anthropic: Trustworthy Agents in Practice][anthropic-shared-resp]
- [Anthropic Engineering: Claude Code auto mode][anthropic-auto]
- [Anthropic Engineering: Making Claude Code More Secure][anthropic-sandboxing]
- [Anthropic Engineering: Scaling Managed Agents][anthropic-managed]
- [Anthropic NIST RFI on Agentic Security][anthropic-nist]
- [Claude Code security documentation][claude-security]
- [Claude Code permission modes](https://code.claude.com/docs/en/permission-modes)
- [MCP Security Best Practices](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices)
- [MCP Authorization documentation](https://modelcontextprotocol.io/docs/tutorials/security/authorization)
- [Kubernetes SIGs Agent Sandbox][k8s-sandbox]
- [Google Cloud: Agent Sandbox on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/agent-sandbox)
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Meta KernelEvolve][kernelevolve]
- [SWE-Bench Verified Leaderboard][swebench]
- [Devin's 2025 Performance Review][devin-review]
- [Goldman Sachs autonomous coder pilot][goldman]
- [Red Hat: Zero trust AI agents on Kubernetes with Kagenti][redhat-kagenti]
- [AIDev: Studying AI Coding Agents on GitHub][aidev]
- [Agentic Workflow Injection in GitHub Actions](https://arxiv.org/abs/2605.07135)
- [Measuring the Permission Gate: Claude Code Auto Mode][permission-gate]
- [Trustworthy Agentic AI Requires Deterministic Architectural Boundaries][deterministic-boundaries]
- [SafeHarness: Security Architecture for LLM-based Agents][safeharness]
- [SandboxEscapeBench: Can AI Agents Escape Their Sandboxes?][sandbox-escape]
- [OWASP Top 10 for Agentic Applications 2026][owasp-agentic]
- [Bishop Fox: The Confused Deputy, MCP Attack][bishopfox-deputy]
- [Docker: MCP Horror Stories, GitHub Prompt Injection][docker-mcp-horror]
- [30+ Vulnerabilities in AI Coding Tools][idesaster]
- [AI Agent Security Incidents Hit 65% of Firms][kiteworks]
- [Bessemer: Securing AI agents in 2026][bessemer]
- [InfoQ: Securing Autonomous AI Agents on Kubernetes][infoq-k8s-agents]
- [AgentSight blog post](agentsight_paper.md)
- [AgentSight repository][agentsight]
- [ActPlane repository][actplane]
