---
date: 2026-09-17
slug: cxl-memory-tier-isolation
title: "Can a Container Keep Its Memory Out of CXL?"
description: "Linux can separate DRAM and CXL NUMA tiers, but cpusets, reclaim demotion, migration, and shared pages still do not guarantee per-container memory residency."
tags:
  - Daily Report
  - Linux
  - CXL
  - Memory Tiering
  - Containers
  - cgroups
research_question: "How should Linux express and verify per-container memory-tier isolation when allocation policy, reclaim demotion, shared pages, and multi-tenant tiering can make different placement decisions?"
source_cutoff: 2026-09-17
status: daily-report
---

# Can a Container Keep Its Memory Out of CXL?

A CXL memory tier often appears to Linux as another NUMA node. That makes a simple deployment rule look plausible: put fast local DRAM in one node, slower CXL memory in another, assign a container only the nodes it may use with `cpuset.mems`, and treat that as a memory-placement boundary.

That rule is useful, but it is not a complete isolation contract.

Linux can move or account pages through paths that are independent of the first allocation. Reclaim can demote pages to a slower memory tier. Changing a cpuset can trigger migration that the cgroup v2 documentation explicitly says may be incomplete. File-backed and other shared pages can have users in several cgroups while still being one physical page. A tiering controller may optimize global hotness or fairness rather than preserve a tenant's original placement intent.

Current Linux CXL documentation makes the problem unusually explicit. Its reclaim guide says older demotion behavior did not respect `cpusets.mems_allowed`. Newer work attempts to respect it, but shared memory instantiated by another cgroup can still be demoted outside the nodes a consumer expected. The documentation therefore warns that `mems_allowed` still cannot provide perfect isolation from remote nodes.

So the hard question is not whether Linux can constrain placement. It can. The harder question is: **what exactly does a container mean when it says that some class of its memory must never reside on a CXL tier, and which kernel paths must preserve that statement?**

<!-- more -->

## CXL allocation policy and residency policy are different contracts

Linux memory placement starts with NUMA policy and the page allocator. With local DRAM and CXL memory exposed as different NUMA nodes, `cpuset.mems` constrains which memory nodes tasks in a cgroup may use. `cpuset.mems.effective` exposes the nodes actually granted after parent and online-node constraints are applied.

That is already more nuanced than a static node list. The cgroup v2 documentation says changing `cpuset.mems` for a populated cgroup attempts to migrate existing pages, but migration may be incomplete and pages can remain behind. It recommends setting memory nodes before spawning tasks when possible.

Tiered memory adds another decision path. When NUMA demotion is enabled, reclaim can choose a lower memory tier instead of immediately swapping or discarding a page. The CXL reclaim documentation describes this as a vmscan decision driven by the kernel memory-tier topology, with tier relationships derived from platform information such as HMAT or CDAT.

A page can therefore see at least two placement decisions during its lifetime:

```text
initial allocation
    -> page allocator / NUMA policy / cpuset

later pressure
    -> reclaim / demotion policy / memory tier topology
```

If those paths do not enforce the same tenant contract, an allocation rule is not a residency rule. A page allocated on CXL because local DRAM was unavailable is different from a page deliberately demoted after cooling, and both are different from a page that stayed on an old node because cpuset migration could not move it.

## Shared pages make ownership a policy question

Private anonymous memory is the easy case. A page normally has one clear memory-cgroup charge and one process family whose performance is affected by its placement.

Shared memory is harder. A file-backed page, shared library page, tmpfs object, or another shared mapping can be physically shared by tasks from different cgroups. Linux still needs one physical residency location. The cgroup that first instantiated or currently owns the charge is not necessarily the only cgroup whose latency or isolation requirement depends on that page.

Suppose container A allows CXL and container B does not. Both execute code from the same shared file page. If the page is attributed to A and reclaim moves it to CXL, B now reads data from a tier its own placement policy would never have selected. Duplicating every shared page per placement domain can restore isolation, but loses capacity and cache sharing. Keeping one copy requires an explicit rule for policy conflicts.

The current CXL reclaim documentation calls out this class of exception directly: shared memory originally instantiated by another cgroup may still be demoted even when demotion attempts to respect `mems_allowed`.

A complete tier-isolation design therefore needs to answer at least three questions:

1. Is the policy attached to the allocating cgroup, the current charger, every active consumer, or the physical page itself?
2. What happens when consumers have incompatible tier policies?
3. When local memory cannot satisfy a hard rule, does the system reclaim, swap, duplicate, throttle, fail allocation, or invoke OOM rather than silently falling back to CXL?

Without those answers, "CXL is not in this container's cpuset" is a useful configuration fact, but not a complete claim about every byte the workload may access.

## Multi-tenant fairness is not the same as hard isolation

Good tiering policy does not necessarily imply isolation.

TPP showed why transparent page placement is useful for CXL systems: keep hot pages in fast local memory, proactively demote colder pages, and preserve enough local headroom for new hot allocations. That is primarily an efficiency and performance objective.

The 2026 Equilibria work goes further for multi-tenant systems. Its production study reports that system-wide tiering can create fairness problems: hotter workloads can capture local memory, later-starting workloads can be disadvantaged, and a thrashing tenant can consume migration work and interfere with neighbors. Equilibria adds per-container tier observability, lower protection and upper bounds for local memory, regulated promotion and demotion, and thrashing mitigation. Its evaluation reports improvements over Linux/TPP of up to 52% for production workloads and 1.7x for benchmarks.

Those mechanisms answer an important question: **how much fast memory should each tenant receive so colocated workloads meet performance objectives?**

Hard tier isolation asks a different question: **may this workload's data ever reside in this tier at all?**

The policies can coexist, but neither implies the other. A fairness controller may intentionally place part of every tenant on CXL. A local-only isolation policy may refuse CXL even if that increases reclaim pressure or OOM risk. One knob should not silently mean both things.

This resembles the earlier [GPU memory-placement report](https://eunomia.dev/research/gpu-memory-placement-evidence/): a placement mechanism needs evidence about what decision was made and whether the resulting placement still matches the intended contract. CXL adds a Linux multi-tenant complication because reclaim, cpusets, cgroups, and shared-page ownership all participate.

## Location counters do not yet prove policy compliance

Linux already exposes useful evidence. `memory.numa_stat` breaks a memory cgroup's footprint down by NUMA node and memory type. NUMA topology and memory-tier sysfs show which nodes belong to which tiers. VM statistics expose migration and demotion activity. Equilibria shows that richer per-container tier counters can make promotion, demotion, and local/CXL usage easier to operate.

But post-hoc residency is not the same as a policy witness.

If an operator finds 2 GiB of a supposedly local-only workload on a CXL node, several explanations are possible:

- the pages were allocated before the cpuset changed;
- migration after the cpuset update was incomplete;
- reclaim demoted pages through a path with different eligibility rules;
- shared pages are charged or owned by another cgroup;
- a tiering controller intentionally moved them;
- node hotplug or topology change altered the effective policy;
- the workload is reading a shared physical page whose accounting identity does not match the consumer being diagnosed.

A node counter reveals the symptom while leaving the policy path uncertain. If the contract is hard isolation, the system also needs to distinguish an allowed exception from a policy escape.

## Where current work is still weak

The first gap is **contract semantics**. Linux exposes cpuset eligibility, cgroup memory control, NUMA policy, tier topology, and reclaim behavior as separate mechanisms. There is no single statement such as "private anonymous pages for this cgroup may use local tier 0 only; shared executable pages may use a remote tier only if every active consumer permits it; otherwise fail closed."

The second gap is **shared-page policy composition**. One physical page can serve consumers with different placement requirements. Charging and allocation ownership are not automatically the right authority for placement policy.

The third gap is **failure behavior**. A hard local-only policy is meaningful only if the kernel defines what happens when local memory cannot satisfy it. Falling back to CXL violates the policy; refusing CXL can cause reclaim, swap, throttling, allocation failure, or OOM. Those outcomes should be selected deliberately.

The fourth gap is **explainability across migration paths**. Current counters show where memory ended up, but a strong isolation claim needs to explain why a page crossed a tier boundary, under which policy generation, and whether the transition was permitted.

Equilibria substantially advances fair multi-tenant CXL placement and observability. The remaining problem here is narrower: turning a best-effort placement preference into a verifiable per-tenant residency contract, including shared pages and pressure behavior.

## Promising directions with academic and production value

### 1. Add a tier-residency hardwall with named failure semantics

A cgroup-level contract could distinguish **allocation preference** from **residency permission**.

For example:

```text
tier_policy:
  allowed_tiers: [local]
  private_anon: hard
  file_private: hard
  shared_file: composed
  on_pressure: reclaim_then_swap
  on_conflict: deny_demotion
  generation: 184
```

The syntax is not the point. Every path capable of changing physical residency would check the same policy generation: initial allocation, reclaim demotion, NUMA migration, explicit migration, cpuset changes, and tiering-controller actions.

A hard policy also needs a defined failure mode. If local memory is full, the system could reclaim another page, swap, throttle the cgroup, return allocation failure where the API permits it, or invoke cgroup-scoped OOM. Silent CXL fallback should be an explicit `best_effort` mode, not an accident.

The artifact would be a Linux prototype attaching a tier mask and pressure behavior to a memory cgroup, with enforcement in allocation and migration eligibility. A container runtime or Kubernetes node agent could translate workload classes into the contract.

Evaluation should run two or more tenants under increasing local-memory pressure across private anonymous, private file-backed, and shared mappings. The primary metric should be **forbidden-residency byte-seconds**: how many bytes spend how long on a disallowed tier. Secondary metrics include reclaim amplification, swap traffic, OOM frequency, tail latency, and stranded CXL capacity.

This mechanism should be rejected if existing cpuset/mempolicy mechanisms, configured before process start, already achieve zero forbidden residency across the same migration and pressure matrix without kernel changes. It should also lose if strict enforcement creates unacceptable failure amplification compared with explicit best-effort placement.

### 2. Treat shared pages as multi-owner placement objects

A second direction is to stop assuming one cgroup charge always identifies the placement authority for a shared physical page.

A prototype could maintain a compact **placement-interest set** for shared folios. It need not enumerate every mapper forever. It could track cgroups with hard tier restrictions or recent active access, using a bounded representation such as a small inline set plus an overflow summary.

When policies agree, placement is straightforward. When they conflict, the kernel needs a composition rule. Candidate rules include hard-deny dominance with the page kept in the intersection of allowed tiers, selective duplication of read-only pages across incompatible placement domains, or an explicit policy-violation budget when remote residency is permitted.

The academic question is whether enough consumer information can be tracked cheaply to improve placement correctness without turning every page into a large multi-tenant metadata object. Production targets include shared libraries, page cache, model weights, and other read-mostly objects on consolidated servers.

Evaluation should use controlled shared mappings where tenants have conflicting tier policies and known access rates. Compare current charge-based behavior, hard intersection, selective duplication, and demand-weighted placement. Measure forbidden residency, duplicate-memory overhead, remote-access latency, migration volume, and metadata cost.

This idea fails if shared-page conflicts are too rare to justify the metadata, or if selective duplication costs more capacity and CPU than accepting remote access with a documented exception.

### 3. Build an adversarial tier-isolation conformance benchmark

A hard contract is difficult to trust without a benchmark designed to break it.

The benchmark should declare a small set of policies and exercise the paths that can invalidate them: allocate before and after a `cpuset.mems` change, force reclaim and NUMA demotion, map the same file into cgroups with conflicting policies, use tmpfs and shared anonymous mappings, vary tenant launch order, trigger tier thrashing, combine zswap with demotion, hot-add or offline memory where supported, and race policy updates with migration.

Each observed physical-page transition should be classified by source tier, destination tier, page class, owner/consumer cgroups, policy generation, and migration reason when available. The oracle then labels the transition as permitted, explicitly degraded, or forbidden.

The useful metrics are policy escapes, maximum escape duration, unexplained migrations, false violation reports, migration overhead, and recovery after policy changes. A controller that is 5% faster but occasionally violates a hard residency rule is not equivalent to one that preserves it.

The academic value is a common failure taxonomy for tier-isolation claims. The production value is qualification before enabling CXL for workloads with data-location, latency, or predictability requirements.

This benchmark is unnecessary if existing kernel selftests and memory-tier tests already cover the same cross-cgroup, shared-page, pressure, and policy-generation failures with a comparable oracle. The first experiment should therefore try to prove that no new counterexample exists.

## A practical rule for current deployments

Today, treat `cpuset.mems` as an important placement constraint, not as proof that every byte a workload can access will remain on those NUMA nodes for its entire lifetime.

Configure memory nodes before starting the workload when possible. Check `cpuset.mems.effective`, not only the requested mask. If NUMA demotion is enabled, understand how the running kernel handles cpuset eligibility in reclaim. Inspect `memory.numa_stat` and tier migration counters for actual residency. Test shared mappings separately from private anonymous memory. Decide in advance whether local-memory exhaustion should permit CXL fallback, swap, throttling, allocation failure, or OOM.

For a performance objective, a fair-share design such as Equilibria's lower protection and upper bound is a better model than pretending every tenant needs a hard wall. For a true isolation requirement, write down the forbidden state and test it under memory pressure.

The earlier [page-level attribution report](https://eunomia.dev/research/page-level-ebpf-memory-attribution/) argued that allocation ownership, RSS, page hotness, and physical-page activity answer different questions. CXL tier isolation exposes the same category error from another angle: the entity charged for a page is not always the complete set of workloads whose policy depends on where that page resides.

## What would change this conclusion?

The conclusion would weaken if Linux provided and documented one end-to-end cgroup memory-tier contract that all allocation, reclaim, migration, shared-page, and hotplug paths were required to preserve, with selftests demonstrating that a hard allowed-tier mask cannot be escaped. In that world, `cpuset.mems` or its successor could legitimately be treated as a residency boundary rather than one input to placement.

It would also weaken if production measurements showed that shared-page and migration exceptions are negligible for workloads that need tier isolation. If setting the cpuset before task creation plus current demotion behavior yields zero meaningful violations across realistic pressure tests, a new hardwall mechanism adds complexity without practical benefit.

Finally, many applications do not need isolation. If the real requirement is an SLO or a guaranteed minimum share of local DRAM, fair-share tiering is the better abstraction. Turning that requirement into a hard "never use CXL" policy can waste capacity and increase OOM risk.

The useful mental model is precise: **NUMA and cpuset controls constrain where memory may be allocated, while tier residency over time is a lifecycle property involving reclaim, migration, sharing, and pressure policy. A hard CXL isolation claim must cover all of those paths, not only the first allocation.**

## References

- Linux kernel documentation, [CXL memory allocation: Reclaim](https://docs.kernel.org/driver-api/cxl/allocation/reclaim.html), accessed 2026-09-17.
- Linux kernel documentation, [Linux CXL early boot and memory tiers](https://docs.kernel.org/driver-api/cxl/linux/early-boot.html), accessed 2026-09-17.
- Linux kernel documentation, [Control Group v2](https://docs.kernel.org/admin-guide/cgroup-v2.html), accessed 2026-09-17.
- Linux kernel documentation, [Page migration](https://docs.kernel.org/mm/page_migration.html), accessed 2026-09-17.
- Kaiyang Zhao et al., [Equilibria: Fair Multi-Tenant CXL Memory Tiering At Scale](https://arxiv.org/abs/2602.08800), 2026.
- Hasan Al Maruf et al., [TPP: Transparent Page Placement for CXL-Enabled Tiered-Memory](https://arxiv.org/abs/2206.02878), 2022/2023.
- Linux kernel source, [torvalds/linux](https://github.com/torvalds/linux), for the implementation paths discussed above.
