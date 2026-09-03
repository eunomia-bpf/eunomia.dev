# Why can reverse-path filtering drop return traffic in an eBPF Kubernetes datapath?

**Short answer:** Linux strict reverse-path filtering validates an incoming IPv4 packet against the kernel's Forwarding Information Base (FIB). If the best route back to the packet's source does not use the interface on which the packet arrived, the kernel drops it. An eBPF datapath may make its forwarding decision from BPF maps, endpoint metadata, or an earlier redirect, but those facts do not automatically appear in the FIB. The packet can therefore be valid according to the CNI datapath and invalid according to `rp_filter` at the same time.

This often appears as a one-way connection: the request reaches a Pod, the Pod emits a SYN-ACK, and the reply is visible on one ingress point of the node but disappears before the expected forwarding or egress point. The durable fix is to align Linux's reverse lookup with the intended path, or to relax source validation only on the narrowly affected interface when asymmetric routing is intentional and another anti-spoofing boundary exists.

## What strict `rp_filter` actually proves

For an IPv4 packet with source `S` arriving on interface `I`, strict mode (`rp_filter=1`) asks whether the FIB's best reverse path to `S` uses `I`. If it does not, source validation fails. Loose mode (`rp_filter=2`) asks only whether `S` is reachable through any interface. Disabled mode (`rp_filter=0`) performs neither check.

The [kernel IP sysctl documentation](https://docs.kernel.org/networking/ip-sysctl.html#rp-filter-integer) defines those three modes and recommends loose mode for asymmetric or otherwise complicated routing. It also contains an easy-to-miss rule: Linux uses the maximum value of `conf/all/rp_filter` and `conf/<interface>/rp_filter`. Setting the interface value to zero does not disable strict filtering if `conf/all/rp_filter` remains one. Some distributions also enable filtering outside the kernel default.

This is a source-address reachability check, not a TCP state check. A valid conntrack entry, a correct SYN/SYN-ACK pair, and a successful eBPF policy decision do not satisfy it. The current kernel implementation returns the explicit drop reason [`SKB_DROP_REASON_IP_RPFILTER`](https://github.com/torvalds/linux/blob/master/net/ipv4/fib_frontend.c) when strict validation cannot match the reverse route to the receiving device.

`net.ipv4.conf.*.rp_filter` applies to IPv4. Do not infer an IPv6 diagnosis or policy from its value; dual-stack paths must be tested independently.

## Why an eBPF CNI can disagree with the FIB

A Kubernetes Service virtual IP is not normally a host with that address. A service proxy captures traffic for the `clusterIP` and port and redirects it to a selected endpoint, as described by the [Kubernetes Service proxy reference](https://kubernetes.io/docs/reference/networking/virtual-ips/). A CNI datapath then has to carry the translated packet to the endpoint and return traffic back toward the client.

An eBPF implementation can know how to do that without installing an equivalent best route in the ordinary Linux FIB. For example, it may:

- map a Pod address to an endpoint or peer device in a BPF map;
- redirect at XDP or TC before the normal IPv4 forwarding decision;
- use a virtual device as a stable attachment point;
- select a policy-routing table with a packet mark; or
- decapsulate or recirculate traffic so that the interface seen by IPv4 input differs from the physical arrival interface.

Now consider a reply whose source is a Pod IP. The eBPF program may have enough metadata to redirect it correctly, while the reverse FIB lookup says that the Pod IP is reachable through a different device, table, or VRF. Strict `rp_filter` trusts the latter and rejects the packet. This is especially easy to trigger when a virtual host-routing device deliberately separates the BPF attachment topology from the route topology.

Hook order matters too. XDP and TC ingress run before the ordinary IPv4 input route lookup. If an early program redirects a packet and it re-enters the stack on another device, source validation evaluates the device visible at that later IPv4 receive point. “The NIC received it” and “`rp_filter` validated it on the NIC” are therefore not equivalent statements.

## A read-only diagnostic sequence

Start by proving the exact disappearance point; do not change `rp_filter` merely because the symptom is asymmetric.

1. **Record the tuple at each boundary.** Capture only the headers needed for diagnosis at the Pod side, its host peer or CNI device, and the expected node egress. Confirm whether the reply source is still the Pod IP and whether DNAT/SNAT has changed either direction. A missing reply at the first host-side point is not an `rp_filter` diagnosis.
2. **Identify the actual receive context.** Redirects, VRFs, and network namespaces can change both the device and the sysctl namespace that matter. Inspect the packet where it enters IPv4 routing, not just on the physical NIC.
3. **Read both effective settings.** In the relevant network namespace, inspect `net.ipv4.conf.all.rp_filter` and `net.ipv4.conf.<ingress>.rp_filter`. The effective mode is their maximum. Also record `src_valid_mark` if policy routing uses a firewall mark.
4. **Reproduce the reverse lookup.** Use `ip route get <packet-source> from <packet-destination> iif <ingress-interface>` with the relevant VRF and mark when applicable. The [`ip route get` manual](https://man7.org/linux/man-pages/man8/ip-route.8.html) explains that `iif` makes the kernel pretend a packet arrived on that interface and resolve the route as the kernel sees it. For strict validation, compare the resolved reverse-path device with the actual ingress device.
5. **Observe a kernel drop reason when supported.** Modern kernels define `IP_RPFILTER` in the [`skb_drop_reason` enum](https://github.com/torvalds/linux/blob/master/include/net/dropreason-core.h). A compatible tracer reading the `skb:kfree_skb` tracepoint can distinguish this from a netfilter, checksum, policy, or BPF-program drop. Check the running kernel's tracepoint format before assuming a particular tool can decode the reason.
6. **Run a bounded comparison.** In an isolated test node or namespace, temporarily compare strict and loose mode on only the implicated ingress interface, with an explicit rollback. If the packet survives and the reverse FIB mismatch is already proven, the causal chain is strong. A successful “disable everything” experiment alone is weak evidence and creates an avoidable spoofing gap.

Repeat the lookup and capture with the packet mark both present and absent if marks select routing tables. By default, the mark is not included in reverse-path lookup. The kernel's [`src_valid_mark` documentation](https://docs.kernel.org/networking/ip-sysctl.html#src-valid-mark-boolean) permits including it when policy routing uses that mark consistently in both directions.

## Fix the disagreement at the right layer

Prefer fixes in this order:

1. **Make the FIB describe the real source path.** Install or correct the per-Pod, prefix, VRF, or policy route so the best reverse lookup for the source resolves through the device on which replies actually arrive. This preserves strict source validation and makes other routing tools agree with the datapath.
2. **Make policy-routing inputs symmetric.** If a mark intentionally chooses the route table, preserve it through the relevant path and enable `src_valid_mark` only after proving that both forward and reverse lookups are supposed to use it. A mark that exists on one direction only can make validation less accurate, not more.
3. **Use loose mode on the specific ingress boundary.** When asymmetry is intentional and the source may legitimately have a different best return interface, loose mode retains a reachability check while allowing that asymmetry. Set and verify both `all` and per-interface values because of the maximum-value rule.
4. **Disable the check only with a replacement trust boundary.** A CNI may already validate endpoint identity, source prefixes, and policy at ingress. If that protection is demonstrably complete, disabling `rp_filter` on a dedicated virtual ingress can be reasonable. Avoid disabling it globally or on unrelated uplinks.
5. **Revisit redirect and attachment topology.** If recirculation makes packets appear on a device that is neither an intended trust boundary nor the FIB's reverse path, changing the attachment or redirect design may be clearer than accumulating sysctl exceptions.

Do not use SNAT as the first workaround merely because it makes the reverse lookup pass. It changes the source identity seen by policy, observability, and applications, and can hide an incoherent route design. Kubernetes documents that ClusterIP traffic within the cluster is normally not source-NATed in iptables mode in its [source-IP behavior guide](https://kubernetes.io/docs/tutorials/services/source-ip/).

## What to test before calling the fix complete

A single same-node TCP handshake is not enough. Exercise at least:

- direct Pod IP and Service ClusterIP destinations;
- same-node and cross-node endpoints;
- local and remote service backends;
- ingress through each CNI-facing device or VRF;
- policy routes with the expected mark, and a deliberately missing mark;
- strict and intended production source-validation settings;
- IPv4 and IPv6 independently in a dual-stack cluster; and
- endpoint replacement, node restart, and route reconciliation.

For every case, assert the packet tuple and ingress device at the validation point, the reverse FIB result, the observed drop reason, and the final application result. This separates four failures that otherwise look identical from the client: service translation, eBPF redirect, reverse-path validation, and later policy or forwarding.

## References

- [Linux kernel documentation: IPv4 `rp_filter` modes and effective-value rule](https://docs.kernel.org/networking/ip-sysctl.html#rp-filter-integer)
- [Linux kernel source: FIB source validation and `IP_RPFILTER` return](https://github.com/torvalds/linux/blob/master/net/ipv4/fib_frontend.c)
- [Linux kernel source: `skb_drop_reason` definition for reverse-path filtering](https://github.com/torvalds/linux/blob/master/include/net/dropreason-core.h)
- [iproute2 manual: resolved FIB lookup with `ip route get`](https://man7.org/linux/man-pages/man8/ip-route.8.html)
- [Kubernetes documentation: virtual IPs and Service proxies](https://kubernetes.io/docs/reference/networking/virtual-ips/)
- [Kubernetes documentation: source IP behavior for Services](https://kubernetes.io/docs/tutorials/services/source-ip/)
- [OpenTelemetry specification: instrumentation libraries, versioning, and stability](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/overview.md)

## Community discussion today

The visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected networking failure appeared within the strict 24-hour window, so the seven-day fallback was not used. This synthesis removes participant, project, employer, and channel identities; message links; exact times; private topology; raw logs; and searchable original wording. No raw transcript was retained, and no reply, reaction, direct message, follow, invitation, or moderation action was performed.

### Return-path validation was the clearest practitioner failure

The strongest in-window troubleshooting report followed a service request through translation and eBPF endpoint routing, then found that the Pod's TCP reply disappeared at the host return path. The practical clue was that the BPF path and ordinary reverse FIB lookup did not agree about the receiving device. The diagnostic sequence above turns that clue into a falsifiable test: locate the exact loss boundary, resolve the reversed tuple with the real ingress context, and confirm the kernel drop reason before changing policy. The original report did not establish that every topology has the same cause, so the conclusion is intentionally limited to paths where those three observations agree.

A second networking project described an eBPF load balancer whose user-space health loop writes a compact backend choice into a map. That design raised a related reliability question: a health bit proves only what its probing layer observed; it does not prove that NAT, return routing, neighbor resolution, and source validation form a working connection. End-to-end tests should therefore assert both directions of a real flow, not only map contents or a backend health endpoint.

### Kernel review concentrated on boundary and lifetime failures

The public kernel development archive was active around bounded iteration of resizable maps, integer overflow in batch and queue operations, socket-reference lifetime, callback and tail-call restrictions, JIT memory checking, and selective module-BTF loading. These topics share a review pattern: fast paths remain safe only when exceptional bounds and ownership transitions are explicit. For network diagnosis, the analogous mistake is to treat “an early eBPF hook returned success” as proof that later IPv4 validation and routing must also succeed.

### Instrumentation maintainers debated beta API coverage

An observability working group discussed whether automatic instrumentation should cover beta APIs while excluding legacy surfaces. The useful distinction is between the instrumented SDK's maturity and the telemetry schema's maturity: supporting a beta SDK call does not require presenting its span shape as a stable contract. A practical implementation can isolate beta patches, gate them by upstream version, fail open when a symbol is absent, and run compatibility tests against explicit version ranges. The unresolved part is upstream-specific: without a published compatibility policy, maintenance cost and breakage frequency must be measured rather than assumed.

The same discussion also flagged bulk-generated work items that had not yet received human review. Generated reports are useful intake, but readiness should be based on a reproduced behavior, a bounded compatibility claim, and an independently reviewable change—not on the volume or confidence of generated text.

### Quiet targets were still checked

Project help and feature areas were empty, quiet, or contained only automated build notifications in the window. The scheduling support areas had no new 24-hour technical question, and one eBPF chat surface had no newer discussion than the already answered tail-call topic. These are accessible-but-quiet findings, not missing coverage reported as zero activity.
