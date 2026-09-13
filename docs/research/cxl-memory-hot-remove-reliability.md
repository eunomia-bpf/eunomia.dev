---
date: 2026-09-13
slug: cxl-memory-hot-remove-reliability
title: "Can Linux Promise That CXL Memory Will Be Removable Later?"
description: "Linux can hot-remove CXL System RAM only after its blocks are evacuated. ZONE_MOVABLE helps, but future removability still depends on runtime ownership."
tags:
  - Daily Report
  - Linux
  - CXL
  - Memory Hotplug
  - Memory Management
research_question: "What evidence and runtime policy are needed to know that CXL memory onlined today can later be evacuated, offlined, and hot-removed without a reboot or unsafe device removal?"
source_cutoff: 2026-09-13
status: daily-report
---

# Can Linux Promise That CXL Memory Will Be Removable Later?

A CXL memory device can be added to a running Linux host, exposed through DAX, and converted into ordinary System RAM. That makes expansion look pleasantly dynamic: add capacity now, let applications use it, and remove the device later when the pool is rebalanced or the hardware needs service.

The last step is the difficult one.

Linux can physically remove memory only after the relevant memory blocks have been offlined. Offlining is not a bookkeeping change. The kernel has to migrate every movable page away, leave no allocation that still depends on the range, and then tear down the software objects that represent the CXL region and device. A memory device that was easy to add can therefore become effectively permanent after the system has been running for hours or days.

`ZONE_MOVABLE` exists largely to improve this situation. It keeps most unmovable kernel allocations out of selected memory so that future offlining has a better chance to succeed. Current CXL documentation goes further: CXL capacity onlined into `ZONE_NORMAL` should be treated as permanently attached to the page allocator, while `ZONE_MOVABLE` is intended to preserve the possibility of hot-unplug later.

But "better chance" is not a removal guarantee. Long-term page pins, huge pages, page-table and memory-map requirements, mixed zone layouts, concurrent allocations, and device-specific mappings can still make evacuation slow or impossible. The old sysfs `removable` attribute does not solve the problem either. Current kernel documentation says it only reports whether the kernel supports memory offlining, not whether this particular block is likely to be removable now.

The practical question is therefore not whether Linux supports CXL hotplug. It is whether a system can **preserve and later prove removability as a lifecycle property**.

<!-- more -->

## Adding CXL memory and removing it are different operations

Linux exposes CXL memory through the DAX subsystem. A CXL DAX region can remain a file-like `/dev/daxN.Y` device that userspace maps directly, or the `dax_kmem` driver can convert it into hotplug memory blocks managed by the ordinary page allocator. The second mode is attractive because applications can use the capacity as System RAM without changing allocators.

That conversion changes the ownership problem. Once CXL capacity becomes System RAM, the kernel and applications can place pages there according to the selected memory zone and normal allocation policy. A later device removal cannot simply revoke those pages. Every dependency on that physical range has to disappear first.

The generic Linux memory-hotplug path reflects this explicitly. Hot-unplug first offlines memory blocks. During offlining, the kernel migrates movable pages and removes free pages from allocation. Only after offlining succeeds can the memory be removed. Kernel documentation also warns that without `ZONE_MOVABLE` there is no guarantee that a block can be offlined successfully, because ordinary kernel zones can contain unmovable allocations such as page tables, `kmalloc()` objects, and other kernel state.

`ZONE_MOVABLE` narrows the allowed allocation classes. Most user-space anonymous pages and page-cache pages are migratable, so they can live there. Most kernel allocations stay in kernel zones. That is a useful structural property, but it creates a second resource constraint: the host still needs enough `ZONE_NORMAL` capacity for kernel metadata and unmovable work. CXL documentation specifically notes that large CXL `ZONE_MOVABLE` pools need sufficient local `ZONE_NORMAL` memory to hold their memory map when `memmap_on_memory` is not used.

This turns one apparent knob into a system tradeoff. Put CXL memory in `ZONE_NORMAL` and the machine can use it broadly, but later removal becomes unreliable. Put it in `ZONE_MOVABLE` and removal becomes more plausible, but the host must reserve enough kernel-zone capacity and avoid workloads whose allocation behavior conflicts with movable memory.

## `ZONE_MOVABLE` is a policy boundary, not a certificate

The generic memory-hotplug documentation is unusually direct about the remaining failure modes. Even with `ZONE_MOVABLE`, offlining can fail in corner cases involving memory holes, mixed zones or NUMA nodes, special blocks, huge pages, and other constraints. Long-term page pinning is particularly important because a page that cannot migrate can hold an entire memory block hostage.

The same documentation describes zone sizing as workload dependent. Too much movable memory can leave the system short of kernel-zone memory even when a large amount of RAM is technically free. Extreme long-term pinning workloads may not work well with `ZONE_MOVABLE` at all. Huge and gigantic pages add further architecture- and configuration-dependent migration rules, and some hot-remove operations can wait indefinitely for a migration target to become available.

This is why a static capability bit is not enough. The property changes as the workload runs.

Consider two identical CXL regions that are both onlined into `ZONE_MOVABLE` at boot. On host A, applications use ordinary anonymous memory and page cache, and no long-lived DMA consumer pins pages in the region. On host B, a device stack or userspace subsystem establishes long-term pins and the machine accumulates large-page state that cannot currently migrate. The two hosts started with the same hardware and kernel capability. Their later hot-remove prospects are different because their runtime state diverged.

A maintenance controller that knows only "CXL hotplug supported" and "region is in `ZONE_MOVABLE`" cannot distinguish those hosts.

## CXL adds another teardown boundary above memory offlining

Memory offlining is necessary but not sufficient for safe physical removal of a CXL device. Current CXL device-hotplug documentation requires careful teardown of the software constructs that manage the device and its memory regions. Hard-removing a CXL.mem device without that teardown is likely to cause a machine check, or at least `SIGBUS` if accesses are limited to userspace.

CXL region topology can also constrain what can be changed dynamically. Firmware reserves CXL fixed memory windows at boot, and regions built from Host-managed Device Memory decoders may need to be torn down and recreated when membership changes. The physical operation therefore sits at the end of a chain:

```text
stop new ownership of the target capacity
        |
        v
migrate or release existing pages and mappings
        |
        v
offline every memory block that belongs to the region
        |
        v
tear down DAX / CXL region and driver state
        |
        v
physically detach or reconfigure the device
```

A failure at any earlier stage should stop the later stage. Treating device removal as a best-effort command after a timeout is unsafe because the failure mode is not merely "capacity stayed online." A physical removal with stale references can become a host fault.

The userspace tooling already exposes some of this reality. `daxctl reconfigure-device` expects System RAM sections to be offline before converting a DAX device back to `devdax`. Its `--force` mode can attempt the offlining, but the documentation warns that overriding auto-online policy may produce a configuration that cannot later be offlined without a reboot. A real ndctl issue shows a CXL device stuck in `system-ram` after one memory section returned `Device or resource busy` during the attempted conversion back to `devdax`.

That is deployment evidence for the same distinction: **successful admission of memory does not imply successful revocation of memory later**.

## Why current observability is not enough

Recent CXL systems work is investing heavily in deciding where memory should live and how memory behavior should be measured. For example, OSDI 2026's NEMO prototypes a telemetry engine on an FPGA-based CXL memory expander and shows that higher-fidelity access evidence can improve tiering and interference decisions. That is useful for placement, but hot-remove needs a different kind of evidence.

A placement controller asks which pages are hot, cold, local, remote, or expensive. A removal controller asks whether every remaining owner of a physical range can be revoked or migrated within an acceptable time. Access frequency is only one input. A cold page can still be unmovable. A page with zero recent accesses can still be pinned by DMA. A low-utilization CXL device can still contain one allocation that prevents the final block from going offline.

This is the same reason the earlier [GPU memory-placement report](https://eunomia.dev/research/gpu-memory-placement-evidence/) separated evidence for a placement decision from a generic page-fault signal. Here the missing evidence is about **evacuability and ownership**, not hotness.

## Where current work is still weak

The first gap is **future-removability admission**. Linux lets an operator choose `ZONE_MOVABLE`, and current automatic online policy tries to balance movable and kernel zones. But there is no application-facing contract that says, "if this region is accepted into service under these workload restrictions, it remains removable within this time bound." The important restrictions, such as long-term pins, huge-page behavior, memory-map placement, and local kernel-zone reserve, are spread across different subsystems.

The second gap is **blocker attribution at removal time**. Offlining can fail or take a long time, but an operator needs to know whether the obstacle is a pin, a huge page, a zone-layout constraint, DAX mapping, kernel metadata, concurrent allocation, or CXL region topology. A Boolean failure from one stage is not enough to decide whether to retry, migrate a workload, reconfigure a device, or schedule a reboot.

The third gap is **end-to-end teardown evidence**. Memory blocks becoming offline does not by itself prove that DAX mappings, CXL regions, decoder state, and driver ownership are ready for physical removal. The boundary crosses memory management and device topology. Current mechanisms expose the individual stages, but production automation still has to compose them correctly.

The fourth gap is **evaluation by realistic blocker class**. A happy-path test that onlines a clean region and immediately offlines it does not model a long-running host. Qualification needs workloads that create page-table pressure, long-term pins, huge pages, DAX mappings, mixed memory pressure, and concurrent allocation before removal is attempted. Otherwise a platform can advertise hot-remove support while regularly falling back to reboot in the field.

## Promising directions with academic and production value

### 1. Preserve a removability budget when the region is admitted

The first direction is to make future removal an explicit admission goal rather than an informal benefit of `ZONE_MOVABLE`.

A small controller could attach a removability policy to each CXL region when it is onlined. The policy would record the target zone, memory-map placement, minimum local kernel-zone reserve, whether long-term page pins are permitted, large-page constraints, the expected maximum evacuation time, and a fallback action if the guarantee cannot be preserved. Allocation paths do not all need to consult a new heavyweight policy engine. The implementation could instead enforce a few high-value boundaries, such as rejecting or redirecting long-term pinning into a removal-reserved region and detecting when local `ZONE_NORMAL` reserve falls below the declared budget.

The artifact could combine a userspace policy daemon with kernel tracepoints, existing sysfs state, and a small kernel change only where a hard admission decision is otherwise impossible. The important output is not "movable memory enabled." It is a versioned region contract that explains what the operator is promising and which runtime events can invalidate that promise.

Evaluation should run anonymous-memory, page-cache, THP/hugetlb, long-term GUP, RDMA/VFIO-like pinning, and memory-pressure workloads across several CXL pool sizes and zone ratios. Compare ordinary `ZONE_MOVABLE`, current automatic online policy, and the admission controller. Measure false-safe decisions, hot-remove success rate, time to evacuate, amount of stranded capacity, application tail latency, and kernel-zone pressure. An ablation should remove pin admission or reserve accounting separately to show which mechanism changes the outcome.

The academic question is whether removability can be treated as a resource property that is conserved through allocation decisions. The production user is a fleet memory-pooling or hardware-maintenance controller that wants hot serviceability rather than a reboot promise.

This direction is not worthwhile if ordinary `ZONE_MOVABLE` plus existing allocation policy already reaches the same removal success rate and evacuation bound, and the extra admission rules never change a deployment decision.

### 2. Produce an evacuation witness before a CXL device can detach

The second direction is a staged proof of quiescence for one region generation.

Before removal, a controller would first prevent new allocations or mappings from entering the target region. It would then enumerate and classify the remaining ownership: ordinary migratable pages, long-term pins, huge pages, DAX mappings, memory-block zone state, and CXL region/device dependencies. After migration and teardown, it would emit a compact witness tied to the exact region generation:

```text
region_generation = G42
new_allocations = blocked
memory_blocks = all offline
remaining_pins = 0
remaining_dax_mappings = 0
cxl_region_users = 0
driver_teardown = complete
physical_remove = permitted
```

The witness is not meant as cryptographic ceremony. Its value is that every transition used by automation has an explicit precondition, and the final detach command can refuse to run against stale evidence from an earlier region configuration.

A prototype could integrate with `daxctl`/`cxl` tooling and existing memory-hotplug interfaces. Evaluation should inject races by starting new mappings, pins, and memory pressure while evacuation runs; reconfigure regions between check and detach; and compare the witness protocol with a script that only waits for memory-block offline state. The main correctness metric is unsafe or stale detach approval. Operational metrics are evacuation latency, false refusal, diagnosis time, and how often the witness identifies a specific blocker rather than returning an opaque busy error.

Academically, this is a cross-subsystem quiescence problem: memory ownership and device topology have to agree on one removal cut. In production, it gives hardware maintenance automation a concrete condition for proceeding without turning "hot-remove" into "try it and hope the host survives."

The idea loses if successful memory offlining plus existing driver removal already forms a complete race-free proof in practice and the witness never catches stale state, hidden ownership, or sequencing errors.

### 3. Benchmark removability debt, not just hotplug success

The third direction is a benchmark that deliberately lets a host age before removal.

Start from a clean CXL region, online it under a declared policy, run a workload phase that creates one blocker class, and then request evacuation. The benchmark should separately exercise:

- ordinary movable anonymous and page-cache memory;
- page tables and other unmovable kernel pressure outside the movable zone;
- long-term GUP or DMA-style pins;
- transparent, huge, and gigantic page configurations;
- direct DAX mappings and `system-ram` conversion;
- concurrent allocations during offlining;
- CXL regions whose device membership or decoder topology requires teardown.

Instead of a single pass/fail result, report a **removability-debt curve** over runtime: how much capacity can still be evacuated, what prevents the remainder, how long evacuation takes, and whether the only remaining recovery action is a reboot. The benchmark can run with QEMU/emulated CXL for broad fault schedules and on real CXL hardware for timing and device behavior.

The strongest comparison is not another synthetic bandwidth test. It is whether competing online policies that look equally healthy at admission produce different removal outcomes after hours of representative activity. A useful suite would also test whether a blocker diagnosis predicts the corrective action: move a workload, release a pin, change huge-page reservation, rebuild a region, or reboot.

The academic value is a lifecycle metric for hotpluggable memory rather than a point-in-time capability test. The production value is qualification: server vendors and operators can decide which CXL configurations are actually serviceable under their workloads.

This benchmark adds little if existing Linux memory-hotplug and ndctl tests already cover the same long-running blocker matrix and accurately predict field removal outcomes at comparable cost.

## A practical deployment rule

If future hot-remove matters, do not treat automatic CXL onlining as a harmless default.

Decide first whether the capacity should remain `devdax` or become System RAM. If it becomes System RAM and later detach is a real requirement, use a `ZONE_MOVABLE`-compatible policy intentionally, retain enough `ZONE_NORMAL` capacity for kernel allocations and memory metadata, and constrain workloads that establish long-lived pins or other non-migratable ownership. During qualification, exercise actual offlining after representative workload phases rather than only after boot.

At removal time, require every memory block in the region to reach the offline state and then complete DAX/CXL region and driver teardown before physical detach. Do not use the legacy `removable` sysfs bit as proof that the current contents can migrate. If evacuation fails, treat the blocker as evidence to diagnose, not as permission to force the hardware operation.

The earlier [GPU checkpoint-recovery report](https://eunomia.dev/research/gpu-checkpoint-recovery-consistency/) asked whether several components really belong to one recoverable application cut. CXL hot-remove has a related lifecycle shape: page ownership, memory-block state, DAX mappings, region topology, and the physical device all have to agree that one resource generation is finished before it disappears.

## What would change this conclusion?

The argument would weaken if Linux gained a hard allocation domain for hot-removable memory that admitted only state with guaranteed migration, bounded evacuation time, and complete blocker attribution, while the CXL stack provided an atomic teardown boundary from page allocator to physical device. In that system, the operator could rely on the kernel contract rather than maintain a separate admission and evacuation protocol.

It would also weaken for deployments that never require live device removal. If a fleet is willing to reboot whenever CXL capacity must be reconfigured or serviced, preserving hot-remove reliability may cost more kernel-zone headroom and policy complexity than it saves.

Finally, experiments could show that the proposed mechanisms are unnecessary. If long-running tests across pinning, huge-page, DAX, pressure, and region-topology workloads find that current `ZONE_MOVABLE`, memory offlining, and CXL teardown already provide predictable removal with clear diagnostics, a new removability contract or witness would add little.

The useful mental model is narrow: **CXL hotplug support tells Linux how memory can enter and leave the machine. Future removability depends on the ownership accumulated while that memory is in service, so it has to be preserved and verified over time.**

## References

- Linux kernel documentation, [CXL Memory Hotplug](https://docs.kernel.org/next/driver-api/cxl/linux/memory-hotplug.html), accessed 2026-09-13.
- Linux kernel documentation, [Memory Hot(Un)Plug](https://docs.kernel.org/admin-guide/mm/memory-hotplug.html), accessed 2026-09-13.
- Linux kernel documentation, [CXL Device Hotplug](https://docs.kernel.org/driver-api/cxl/platform/device-hotplug.html), accessed 2026-09-13.
- Linux kernel documentation, [CXL DAX Driver Operation](https://docs.kernel.org/next/driver-api/cxl/linux/dax-driver.html), accessed 2026-09-13.
- pmem/ndctl, [`daxctl reconfigure-device`](https://github.com/pmem/ndctl/blob/main/Documentation/daxctl/daxctl-reconfigure-device.txt), accessed 2026-09-13.
- pmem/ndctl issue #256, [CXL device cannot be changed from system-ram mode to devdax mode](https://github.com/pmem/ndctl/issues/256), opened 2023-10-09, accessed 2026-09-13.
- Shihang Li et al., [Finding NEMO: Nimble and Expressive Memory Observability](https://www.usenix.org/conference/osdi26/presentation/li-shihang), OSDI 2026.
