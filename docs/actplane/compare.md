# ActPlane Compared with Existing Products

This page positions ActPlane against products that teams already use for AI
guardrails, sandboxes, observability, and runtime security. It is not a buying
guide. Product capabilities change, so treat the matrix as a public-docs
snapshot and verify current vendor documentation before procurement.

ActPlane's control point is different from most AI safety products: it watches
the operating-system behavior of an agent process tree. That makes it
complementary to model guardrails, tool permissions, sandboxes, and LLM
observability.

## Safety, Security, and Compliance

Use these terms precisely in customer-facing material:

| Term | What ActPlane means | Typical controls | Boundary |
| --- | --- | --- | --- |
| Safety | Keeping agent behavior inside approved operating rules | read-only review agents, test-before-commit gates, prompt-injection review gates, workspace write limits | Not content moderation, jailbreak detection, or a guarantee about model intent |
| Security | Protecting files, processes, and network effects at the OS boundary | secret no-exfil, production database mediation, blocked destructive file operations, restricted outbound connections | Not a VM escape boundary, full DLP product, complete firewall, or EDR replacement |
| Compliance and governance | Making agent controls reviewable, repeatable, and auditable | policy-as-code, `compile --explain` reports, CI support reports, approval metadata for runtime deltas, violation feedback records | Supports control evidence; does not certify SOC 2, ISO 27001, HIPAA, or any legal/regulatory outcome by itself |

The concise positioning is: ActPlane provides OS-level safety, security, and
governance controls for AI agents. For hard security claims, verify the relevant
clauses with `actplane compile --explain` and confirm BPF-LSM support with
`actplane doctor`.

## Product Comparison Matrix

Legend:

- `Native`: advertised or documented as a primary product capability.
- `Partial`: adjacent capability or possible integration point, but not the
  product's main documented enforcement boundary.
- `No`: not the documented control point.

| Product | Category | Primary control point | Content / prompt safety | Sandbox isolation | OS-level deny | Agent process-tree policy | Derived data across files/processes | Workflow gates | Agent corrective feedback | Governance evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ActPlane | Agent OS policy harness | Process, file, network, labels, and temporal gates for agent subprocesses | Partial | No | Native | Native | Native | Native | Native | Native |
| [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) | Managed model guardrails | Model inputs and outputs in Bedrock applications | Native | No | No | No | No | Partial | Partial | Partial |
| [Azure AI Content Safety Prompt Shields](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection) | Managed content safety and prompt-attack detection | User prompts and document inputs | Native | No | No | No | No | Partial | Partial | Partial |
| [NVIDIA NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/home) | Application guardrail framework | LLM app input/output flows | Native | No | No | No | No | Partial | Partial | Partial |
| [Check Point / Lakera AI Guardrails](https://docs.lakera.ai/docs/prompt-defense) | AI application security | Prompts, tool responses, tool descriptions, and app traffic | Native | No | No | No | No | Partial | Partial | Partial |
| [Pangea AI Guard](https://pangea.cloud/docs/ai-guard) | AI application security API | AI app traffic, prompt injection, PII, malicious content | Native | No | No | No | No | Partial | Partial | Native |
| [Cloudflare AI Gateway](https://www.cloudflare.com/products/ai-gateway/) | AI gateway | Provider routing, caching, logs, metrics, rate limits, guardrails | Native | No | No | No | No | Partial | No | Native |
| [Portkey Guardrails](https://portkey.ai/docs/product/guardrails) | AI gateway guardrails | Input/output guardrail checks on gateway requests | Native | No | No | No | No | Partial | Partial | Native |
| [LiteLLM Proxy Guardrails](https://docs.litellm.ai/docs/proxy/guardrails/quick_start) | AI gateway guardrails | Guardrails on proxy requests, including per-key controls and traces | Native | No | No | No | No | Partial | Partial | Native |
| [TrueFoundry AI Gateway](https://www.truefoundry.com/docs/ai-gateway/tfy-prompt-injection) | AI gateway guardrails | Managed prompt-injection and jailbreak checks on gateway traffic | Native | No | No | No | No | Partial | Partial | Native |
| [E2B](https://e2b.dev/docs) | Agent sandbox | Isolated cloud sandboxes for code, data, and tools | No | Native | Partial | Partial | No | No | No | Partial |
| [Daytona](https://www.daytona.io/docs/en/) | Agent sandbox | Isolated sandbox computers with kernel, filesystem, network, CPU/RAM | No | Native | Partial | Partial | No | No | No | Partial |
| [LangSmith](https://docs.langchain.com/langsmith/observability) | LLM observability and evaluation | Traces, production metrics, debugging, evaluations | Partial | No | No | No | No | Partial | No | Native |
| [Langfuse](https://langfuse.com/docs) | LLM observability and evaluation | Traces, costs, latency, prompts, evaluations | Partial | No | No | No | No | Partial | No | Native |
| [Tetragon](https://tetragon.io/) | eBPF runtime security | Kubernetes-aware process, file, network, and kernel events | No | No | Native | Partial | No | No | No | Native |
| [KubeArmor](https://kubearmor.io/) | Runtime security enforcement | Workload hardening with eBPF and Linux Security Modules | No | Partial | Native | Partial | No | No | No | Native |

## How to Read the Matrix

The model-guardrail and AI-gateway products are strongest before or after the
LLM call. They are the right layer for harmful content, denied topics,
prompt-injection detection, PII redaction, routing, rate limits, and app-level
policy. They are not OS controls: if an agent reaches a shell, generated script,
package manager, or SDK path, the guardrail sees only what the application sends
through it.

The sandbox products are strongest when the priority is isolating code
execution. They give the agent a separate environment for files, tools, and
commands. They do not by themselves express workflow rules such as "tests must
pass after the last edit" or data-flow rules such as "anything derived from
`.env` cannot later reach the network."

The observability products are strongest for understanding runs. They collect
traces, costs, latencies, prompts, responses, evaluations, and debugging
metadata. They are usually after-the-fact or application-instrumented; they do
not make OS side effects impossible.

The runtime security products are closest to ActPlane at the kernel boundary.
Tetragon and KubeArmor can enforce process, file, and network policies for
hosts, pods, containers, and workloads. ActPlane differs by making the AI agent
subtree the policy subject, adding derived-data labels, temporal workflow gates,
and corrective feedback that agents can use to recover.

## Where ActPlane Is Distinct

ActPlane should lead when the control has to survive common bypass paths:

- `git push` is forbidden whether called directly, through a shell, or through a
  generated Python script.
- A secret read by one process must not be sent out later by another process.
- A production database file must only be opened through the migration tool.
- A coding agent must not commit until tests have passed after the last edit.
- A review subagent must stay read-only for its whole descendant process tree.
- A release artifact must not be published until a review step endorses the
  session.

## Where Another Product Should Lead

Use another control as the primary boundary when:

- You need harmful-content moderation, jailbreak detection, routing, rate
  limits, or gateway-level policy before the model call. Use Bedrock
  Guardrails, Azure AI Content Safety, NeMo Guardrails, Lakera, Pangea,
  Cloudflare AI Gateway, Portkey, LiteLLM, TrueFoundry, or a similar AI
  guardrail/gateway layer.
- You are running untrusted generated code and need a separate execution
  environment. Use E2B, Daytona, a container, or a VM.
- You need run review, prompt debugging, cost tracking, evaluations, or product
  analytics. Use LangSmith, Langfuse, or your existing observability stack.
- You need broad Kubernetes or host workload hardening. Use Tetragon, KubeArmor,
  EDR, or platform security controls.
- You need organization-wide content inspection or legal compliance management.
  Use dedicated DLP, governance, identity, approval, and compliance systems.

ActPlane can still be paired with those layers, but it should not be presented
as their replacement.

## Deployment Pairings

| Pairing | What the other product does | What ActPlane adds |
| --- | --- | --- |
| Bedrock / Azure / Lakera / Pangea + ActPlane | Screens prompts, outputs, PII, unsafe content, or prompt attacks | Controls downstream OS side effects after the agent acts |
| Cloudflare / Portkey / LiteLLM / TrueFoundry + ActPlane | Routes model traffic, applies gateway guardrails, records traces, and manages access | Covers shell, subprocess, generated-code, file, and network effects outside the gateway path |
| NeMo Guardrails + ActPlane | Defines app-level conversational and tool-flow policies | Enforces file, process, network, data-flow, and temporal rules below the app |
| E2B / Daytona + ActPlane | Gives the agent an isolated execution environment | Adds policy inside the environment: lineage, workflow freshness, and feedback |
| LangSmith / Langfuse + ActPlane | Records traces, evaluations, costs, and debugging context | Stops selected actions during the run and emits violation reasons |
| Tetragon / KubeArmor + ActPlane | Provides broad host, pod, or workload runtime security | Adds agent-specific process-tree labels, workflow gates, and recovery feedback |

## Non-Goals

ActPlane is not:

- a VM escape boundary
- a full DLP or content-inspection system
- a complete network firewall
- a replacement for enterprise identity and approval systems
- a substitute for human review of high-risk production changes
- a generic runtime security platform for all host workloads

ActPlane is best described as an OS-level policy harness for AI agents: it makes
selected system actions deterministic, reviewable, and enforceable across the
agent's actual process tree.
