---
date: 2026-08-27
title: "Can eBPF Preserve Complete Mediation Across Host and Offload Paths?"
description: "When eBPF policy spans host networking, SmartNIC fast paths, and DPU offload, every reachable packet path still needs a policy-equivalent enforcement point."
tags:
  - Daily Report
  - eBPF
  - Networking
  - Security
  - SmartNIC
  - DPU
  - XDP
  - Offload
research_question: "How can an eBPF network security policy guarantee that every reachable packet path is mediated by an equivalent policy generation when enforcement may run in host software, SmartNIC hardware, or a DPU and paths can change during offload or fallback?"
source_cutoff: 2026-08-27
status: daily-report
---

# Can eBPF Preserve Complete Mediation Across Host and Offload Paths?

A host starts with one network security policy in software. Later, an operator enables SmartNIC or DPU offload to save CPU. Packets that match hardware rules can now stay on the device fast path, while misses may travel through a representor and return to the host slow path. A reset, unsupported rule, or mode change can move traffic between those paths again.

The security requirement did not change. Every packet that should be constrained by the policy still needs to cross an enforcement point that implements the right policy generation.

That sounds obvious, but attachment state alone does not prove it. A BPF program can be loaded successfully on the host while some traffic is switched in hardware. A rule can be offloaded successfully while another path falls back to software. The device and host can even support the same high-level action while exposing different helpers, maps, metadata, and update boundaries.

<!-- more -->

This report argues that **complete mediation for heterogeneous eBPF networking should be expressed as a path-coverage invariant**. For each reachable packet-path class and policy generation, the system should be able to identify at least one trusted enforcement point that sees the required context, uses policy-equivalent state, and has a defined fallback when that enforcement point disappears.

This is narrower than the earlier report on [where eBPF should execute in a heterogeneous system](https://eunomia.dev/research/heterogeneous-ebpf-execution-placement/). That report asks how a planner chooses among semantically valid backends. Here the backend has already been chosen. The question is whether the resulting topology still mediates every path that matters to security.

It is also different from [authorization revocation](https://eunomia.dev/research/ebpf-authorization-revocation/). Revocation asks how long a previously valid authorization can survive in cached state after it becomes invalid. Complete mediation asks a more basic question: did the packet cross an enforcement point that could apply the current authorization rule at all?

## Offload already creates more than one packet path

Linux representors make the split explicit. The kernel documentation describes a network-function representor as both a control-plane handle and a **slow path for traffic that does not hit an offloaded fast-path rule**. Rules such as TC filters can be installed through the representor and offloaded to hardware. The design goal is that packet behavior, apart from performance, should be the same whether a rule is executed in software or hardware. That goal already implies two execution paths whose equivalence matters to policy correctness. See the Linux [Network Function Representors](https://docs.kernel.org/networking/representors.html) documentation.

The device capabilities are not uniform either. Linux exposes per-netdev XDP feature flags, including whether a device supports basic XDP actions, redirect, AF_XDP zero copy, and hardware offload. The current [`netdev` generic-netlink specification](https://docs.kernel.org/netlink/specs/netdev.html) therefore gives userspace enough information to learn that two interfaces can have different XDP execution capabilities.

Cilium's current BPF toolchain documentation makes the operational distinction concrete. XDP can run as native driver XDP, generic XDP, or SmartNIC hardware offload. Hardware offload can execute the program directly on the card, but it does not expose every BPF map type or helper available to native XDP. Cilium also notes that selecting generic `xdp` can fall back from native to generic mode, while explicitly selecting `xdpdrv` can require native execution and fail instead of accepting that fallback. Switching between XDP modes is not atomic. These are useful controls, but they describe **where one XDP program runs**, not whether an entire multi-path network topology remains covered during and after the change. See Cilium's [BPF development toolchain](https://docs.cilium.io/en/stable/reference-guides/bpf/toolchain/).

Research systems show why offload is worth pursuing. [hXDP](https://www.usenix.org/conference/osdi20/presentation/brunella) demonstrated XDP execution on an FPGA NIC and reported much lower forwarding latency than its CPU baseline on the evaluated workloads. The performance case for moving policy closer to packets is real. The security problem is what the control plane must prove before it treats that faster path as equivalent to the slower one.

## Complete mediation is a coverage property, not a placement score

A placement planner can compare latency, CPU use, state crossings, and device capability. A security policy needs another answer first: **which packet paths are reachable, and which enforcement point covers each one?**

Consider one host with a physical uplink, two VFs, a representor for each VF, an eSwitch fast path, and a host software path. A policy update might affect at least these path classes:

- uplink to VF through a hardware rule;
- uplink to VF after a hardware miss and representor slow path;
- VF to uplink through hardware or host fallback;
- VF to VF or hairpin traffic that can stay inside the device switch;
- traffic redirected through XDP or AF_XDP;
- packets observed while an offload rule is being replaced, withdrawn, or restored after a device reset.

The exact list is device- and deployment-specific. That is the point. Loading a BPF object on `eth0` is not a proof that all of these classes cross `eth0` in the same way.

A useful complete-mediation contract should therefore describe a **path class** by the facts that determine where enforcement can happen: source port or function, destination class, direction, switching domain, offload mode, and relevant fallback. For each policy generation `g`, every reachable path class `p` should map to at least one enforcement point `e` for which the system can establish:

1. `e` is active for generation `g`;
2. `e` observes the context required by the policy;
3. its verdict semantics are equivalent to the intended policy for that path;
4. the state it reads is either authoritative or within an explicitly allowed consistency bound;
5. losing `e` moves the path to another covered enforcement point or stops the path from carrying policy-sensitive traffic.

This is intentionally not a requirement to duplicate the same BPF bytecode everywhere. A SmartNIC rule, a host TC program, and an XDP program can implement different compiled forms. What must remain equivalent is the security decision for the covered path and policy generation.

The contract also does not require logging every packet. The control plane can prove most of the topology statically or at update time, then retain compact generation and coverage evidence for later diagnosis. Per-packet witnesses are useful for testing and sampled audit, not necessarily for the production hot path.

## Why ordinary deployment success is not enough

A successful load or offload answers a local question: one program or rule was accepted by one backend. Complete mediation is a global property over reachable paths.

A simple failure illustrates the difference. Suppose a deny rule is successfully offloaded for uplink-to-VF traffic. The host reports the new generation as ready. VF-to-VF traffic, however, can be switched by a separate hardware path that the controller did not include in its coverage model. Both local operations succeeded, yet the global policy is incomplete.

The reverse failure is possible during fallback. A device loses an offloaded rule and begins sending misses to the representor. If the host slow path already has an equivalent policy generation, the fallback can be safe. If the host copy is one generation behind, the same packet can receive a different verdict solely because the hardware state changed.

This is closely related to the authority provenance problem from [multi-tenant network policy composition](https://eunomia.dev/research/ebpf-network-policy-composition/), but it is not the same problem. Composition determines which owner and rule should decide a packet. Complete mediation determines whether every reachable path reaches an enforcement point capable of applying that composed decision.

## Where current work is still weak

### Attachment and offload status do not prove topology coverage

Linux can tell userspace whether a netdevice supports XDP hardware offload, and control planes can observe whether individual programs or flow rules were accepted. Representors expose the slow path and act as handles for offloaded switching rules. These are necessary pieces of information.

The missing element is a common assertion that connects the packet topology to those attachments: *all reachable path classes for policy P at generation G are covered*. Without that assertion, operators must infer complete mediation from a set of per-device states.

The gap is material if topology changes, SR-IOV configuration, hairpin paths, or offload rules can create a reachable class that bypasses every intended enforcement point. A useful test is a topology mutation benchmark that repeatedly changes those conditions and compares the controller's claimed coverage with ground-truth packet paths.

### Software and hardware verdict equivalence is usually assumed locally

Linux representor semantics aim for the same behavior whether a TC rule is offloaded or handled in software. XDP offload similarly starts from a program that has been accepted for device execution. But a larger policy can depend on map state, metadata, helper behavior, or generation boundaries that are not identical across backends.

The missing capability is not a universal proof that two arbitrary BPF programs are equivalent. A narrower and more deployable contract can require each compiled backend artifact to declare the policy operations and state version it implements, then test equivalence on the packet classes that backend is allowed to cover.

This matters because silent disagreement is worse than an explicit unsupported placement. A differential test should feed the same policy-relevant packets and state generations through software and offloaded implementations and fail activation when their externally visible verdicts differ outside an allowed boundary.

### Fallback and reset have no complete-mediation latency metric

Offload systems need to handle rule replacement, device reset, driver reload, capability loss, and explicit mode changes. Existing mechanisms can provide software fallback or reinstall hardware state, but the security question is temporal: during the transition, was there any interval in which a reachable path had no current enforcement point?

The missing measurement is a **coverage gap**, not only recovery time. A benchmark should timestamp the last packet accepted by an obsolete or uncovered path and the first packet handled by a valid replacement. A system with a fast recovery time can still violate complete mediation if it accepts one unauthorized packet in the middle.

## Promising directions with academic and production value

### 1. Compile reachable paths into an enforcement-coverage plan

**Gap.** Controllers know interfaces, representors, XDP capabilities, and individual offload state, but they rarely turn that information into an explicit proof obligation over all policy-relevant packet paths.

**Mechanism.** Build a small topology compiler that discovers or receives the host ports, VFs/SFs, representors, eSwitch relationships, routing/redirect domains, and available BPF/offload capabilities. It enumerates policy-relevant path classes and assigns each class to one or more enforcement points. Activation fails if any reachable class has no enforcement point with the required context, state version, and verdict capability.

The compiler should distinguish **coverage** from optimization. A separate placement layer may choose the cheapest eligible enforcement point, but it can choose only inside the set that satisfies the coverage contract.

**Delta.** The [heterogeneous placement report](https://eunomia.dev/research/heterogeneous-ebpf-execution-placement/) proposes a target manifest and planner for selecting among valid execution locations. This direction adds a different invariant: the selected locations together must cover the reachable network-path graph. One perfect backend is insufficient if traffic can route around it.

**Artifact.** A public path-model schema, Linux netlink/devlink adapters, support for representor and XDP capability discovery, and a checker that emits both the selected coverage map and a human-readable explanation of uncovered paths.

**Evaluation.** Use physical and virtual topologies with uplink, VF/SF, representor, host, XDP, and hardware-offload paths. Mutate routing, offload support, VF relationships, and redirect rules. Compare the checker with per-device attachment inspection and with a host-only policy baseline. Measure uncovered-path detection, false alarms, plan computation time, and throughput cost of the chosen enforcement plan.

**Academic value.** The general question is how to prove complete mediation when enforcement is distributed across programmable network locations rather than one reference monitor.

**Production value.** A CNI, service-mesh datapath, firewall controller, or SmartNIC management plane could reject a rollout that leaves one path unmediated instead of discovering the gap during an incident.

**Failure condition.** If real deployments have a small fixed set of paths that existing kernel and device APIs already cover unambiguously, and topology fuzzing finds no hidden classes, a general compiler adds unnecessary complexity.

### 2. Make policy generations continuous across offload and fallback

**Gap.** A path can stay mediated yet switch from hardware to software while the two enforcement points run different policy generations.

**Mechanism.** Give each enforcement domain a compact status tuple: policy identity, generation, covered path classes, backend mode, and readiness. A rollout barrier becomes satisfied only when every required path class has at least one ready enforcement point for the target generation. If a device loses an offloaded rule, the controller may expose the path to software only when an equivalent host generation is already active; otherwise it quiesces or fails closed for policy-sensitive traffic.

This can be implemented without synchronously deleting every old rule. The key property is that a packet never moves from one enforcement point to another with an untracked generation downgrade.

**Delta.** The [authorization revocation report](https://eunomia.dev/research/ebpf-authorization-revocation/) tracks when stale authority stops being reusable inside persistent state. This direction tracks continuity of the enforcement point itself as traffic moves between hardware and software paths.

**Artifact.** A coordinator that consumes BPF link/program state plus netdev/devlink/offload status, exposes per-path generation readiness, and drives a two-phase activation or fail-closed fallback protocol.

**Evaluation.** Repeatedly update policy while forcing hardware rule eviction, NIC reset, representor fallback, controller restart, and XDP mode changes. Compare best-effort offload, host-only enforcement, and the generation barrier. Measure unauthorized packets, packets evaluated by the wrong generation, recovery time, update latency, and steady-state packet cost.

**Academic value.** The mechanism turns a vague "fallback is safe" assumption into a temporal safety property over changing enforcement topology.

**Production value.** Operators gain a concrete readiness signal for upgrades and device faults instead of trusting that software and hardware convergence happened in the right order.

**Failure condition.** If current control planes already keep hardware and software generations synchronized under all tested fault transitions, or if host-only failback is cheap enough to make a distributed barrier unnecessary, the simpler design should win.

### 3. Benchmark policy escapes, not only offload throughput

**Gap.** Offload evaluations usually ask how much CPU or latency a device saves. Security evaluations often assume the tested packets traverse the intended enforcement point. Neither setup directly measures whether traffic can escape mediation when path selection changes.

**Mechanism.** Build a ground-truth path-escape harness. The harness generates packet classes whose intended verdicts are known, forces them through hardware hits, hardware misses, representor slow paths, VF-to-VF/hairpin paths, XDP modes, and failure transitions, then records the actual path and policy generation that handled each packet.

The primary outcome is not packets per second. It is the count and timing of packets that were accepted without a valid enforcement witness, accepted by an obsolete generation, or received a verdict different from the policy reference.

**Delta.** hXDP and other offload work establish the performance value of device-side execution. This benchmark keeps that performance dimension but adds a correctness workload specifically designed to make incomplete mediation visible.

**Artifact.** A reproducible topology, traffic generator, fault injector, reference policy interpreter, and trace format that can be used by kernel, CNI, SmartNIC, and DPU implementations.

**Evaluation.** Run the same policy under host-only enforcement, best-effort offload, and coverage-aware rollout. Hold packet load and policy update schedule fixed. Report throughput and CPU alongside policy escapes, stale-generation verdicts, detection latency, and recovery behavior. Include simple topologies where all systems should tie, so the new mechanism is penalized when it adds overhead without preventing a real failure.

**Academic value.** The benchmark tests a general systems-security property that conventional throughput and rule-install metrics do not capture.

**Production value.** Vendors and operators can test whether a claimed hardware acceleration path preserves the security contract through updates and faults, not only in steady state.

**Failure condition.** If multiple independent implementations produce zero policy escapes across adversarial path transitions and the coverage-aware mechanisms do not improve detection or recovery, complete mediation may already be adequately enforced by existing abstractions.

## What would change this conclusion?

The argument depends on heterogeneous deployments having reachable packet paths whose enforcement status cannot be inferred safely from one attachment or offload result. It also assumes that hardware and software policy generations can diverge during realistic updates or faults.

Strong evidence against the proposed mechanisms would be a production-grade control plane that already exports a complete path-to-enforcement map, proves policy-equivalent hardware/software generations, and survives topology changes, offload withdrawal, and device reset with zero uncovered or stale-generation packets under an adversarial benchmark. In that case, the missing work would be standardization and measurement rather than a new runtime protocol.

A second boundary is cost. If the topology is simple enough that operators can keep one host enforcement point on every policy-relevant path without meaningful performance loss, distributed coverage coordination is the wrong solution. Complete mediation is the property to preserve; heterogeneous enforcement is only worthwhile when it buys enough performance or locality to justify the extra state.

The most useful next experiment is therefore not another peak-throughput comparison. It is to take one real host-plus-SmartNIC or host-plus-DPU deployment, enumerate every reachable packet path, force policy updates and device failures, and ask a binary question for every accepted packet: **which current enforcement point authorized this packet, and can the system prove that no reachable path existed without one?**

## References

- Linux Kernel documentation, [Network Function Representors](https://docs.kernel.org/networking/representors.html).
- Linux Kernel documentation, [`netdev` generic-netlink specification](https://docs.kernel.org/netlink/specs/netdev.html).
- Cilium documentation, [BPF development toolchain and XDP operation modes](https://docs.cilium.io/en/stable/reference-guides/bpf/toolchain/).
- Brunella et al., [hXDP: Efficient Software Packet Processing on FPGA NICs](https://www.usenix.org/conference/osdi20/presentation/brunella), OSDI 2020.
- Eunomia Daily Report, [Where Should eBPF Run in a Heterogeneous System?](https://eunomia.dev/research/heterogeneous-ebpf-execution-placement/).
- Eunomia Daily Report, [How Long Can a Revoked Authorization Stay Alive in an eBPF Datapath?](https://eunomia.dev/research/ebpf-authorization-revocation/).
- Eunomia Daily Report, [How Should eBPF Compose Multi-Tenant Network Policies?](https://eunomia.dev/research/ebpf-network-policy-composition/).
