# Why does a Cilium LoadBalancer frontend stay missing after the conflicting Service that owns a shared VIP port is deleted, and how should a controller recover it?

The frontend is not broken — it was never installed. Cilium's kube-proxy replacement validates a service's frontends atomically, and the moment the second service claimed a (VIP, port, proto) that another service already owned, the entire upsert was rejected: `ErrFrontendConflict` fires in `validateFrontends` before the service row is inserted, so the existing service keeps its working frontends and the claimant's new port is absent from the BPF maps. When the owning service is deleted, the reflector's delete handler removes only the owning service's frontends and row; it has no path that re-queues or re-processes the rejected claimant, and upsert errors are reported to the reflector's degraded health state and never retried. The claimant recovers only when a new upsert event for it arrives (for example, a Kubernetes update that produces a non-empty patch). This is a reconciliation gap, not documented behavior. The supported way to migrate a port off a shared VIP is to order the handoff — release the source service's port before the target claims it, or guarantee a follow-up claimant event — and to treat `Frontend.Status` reaching `done`, not desired-state acceptance, as the convergence signal.

## The mechanism

Cilium's kube-proxy replacement holds each LoadBalancer service's frontends in BPF-backed state tables and reconciles them from Kubernetes service and endpoint events. The write path is `pkg/loadbalancer/writer/writer.go`:

- `UpsertServiceAndFrontends(txn, svc, fes...)` first runs `validateFrontends(txn, fes...)` on the full requested frontend set, before it touches the service table with `w.svcs.Insert(txn, svc)`.
- `validateFrontends` walks each frontend address; if that address is already owned by a different service, it returns `ErrFrontendConflict` wrapped as `frontend already owned by another service: <vip:port/proto> is owned by <ns/name>`.
- On that error the whole upsert returns immediately: no service-row insert, no new frontend writes, no backend release. The owning service's existing frontends stay untouched in the BPF maps.

That is the "frontend stays missing" state: the claimant's service row keeps its old port set (its surviving frontends keep working), and the conflicted port was never created.

The read side is `pkg/loadbalancer/reflectors/k8s.go`. Its service-event handler has two arms:

- `resource.Upsert` converts the Service into frontends plus backends and calls `UpsertServiceAndFrontends`; any error is recorded through the reflector's health object (`rh.update` → degraded status plus a warn log). There is no retry.
- `resource.Delete` calls `DeleteServiceAndFrontends(txn, name)`, which removes the deleted service's frontends and then its row. It does not re-evaluate any other service against the freed addresses, and it does not re-queue a claimant whose earlier upsert was rejected.

Recovery therefore requires a new upsert event for the claimant. A Kubernetes Update that produces a non-empty patch (an annotation flip is the minimal one) generates that event; an empty-patch apply does not.

Event coalescing explains the ordering failure. `k8s.go` combines service and endpoint events with `stream.Buffer` at a 500-event / 500ms window into an `InsertOrderedMap`. The container's `InsertOrderedMap.Insert` document says "An update will not affect the ordering", so within one batch events process in first-arrival order. If a claim event arrives before the release event, the claim is processed first (rejected, and never retried), then the release frees the frontend, and nothing re-runs the claim. The release and the claim can sit inside the same coalescing batch, so "the owner was deleted" does not imply "the claim was reprocessed after the release."

`Frontend.ID` (the BPF service-map key) is documented in `pkg/loadbalancer/frontend.go` as valid only once `Frontend.Status` is `done`, and `cilium service list` exposes `Status`, `Since`, and `Error` per frontend. That is the supported datapath-convergence signal.

## Verification and debugging path

Reproduce and confirm the rejection-before-write sequence against the public sources:

1. With two services on the same VIP via `lbipam.cilium.io/sharing-key`, add a port to the second service that the first owns. The claimant's upsert logs `frontend already owned by another service: <vip:port> is owned by <ns/name>` (`ErrFrontendConflict`); the claimant's frontend does not appear in `cilium service list`, while the owner's existing frontends keep serving.
2. Delete the owning service. `DeleteServiceAndFrontends` runs for the owner; the freed frontend does not automatically appear under the claimant, because the rejected claimant was not re-queued.
3. Generate a follow-up upsert for the claimant (a non-empty-patch Update). The frontend is installed; `Frontend.Status` transitions from pending to done, and the ID is valid.
4. To see the coalescing boundary: batch a claim and the corresponding release into a single ~500ms window and confirm the claim processes before the release (first-arrival order), and that a claim arriving before the release leaves the frontend missing until the next claimant event.
5. Check convergence with `Frontend.Status` = done (via `cilium service list`), not with the Kubernetes Service object being accepted: desired-state acceptance is not datapath convergence.

## The limitation

The gap is that the reconciler has no ownership handoff: deleting the owning service does not re-queue the rejected claimant. This is not documented as a supported migration path, and no released fix was found — `ErrFrontendConflict` is still present in current public Cilium versions. The supported practice is ordering, not retry: release the source service's port before the target claims it, or guarantee a follow-up claimant event (an annotation-only Kubernetes update is the minimal non-empty-patch trigger, but that is an incidental implementation behavior, not a documented API). Never hold two services to the same (VIP, port).

The hard limit is the K8s API completion of the source delete: it does not guarantee the informer delivered and processed the release before the claimant's upsert, and the 500-event/500ms coalescing window can absorb a release and re-deliver it out of order. No single-event check can prove the datapath is safe; the only honest convergence signal is `Frontend.Status` reaching done.

## References

- [Cilium v1.18.2 — `pkg/loadbalancer/errors.go`](https://github.com/cilium/cilium/blob/v1.18.2/pkg/loadbalancer/errors.go) — `ErrFrontendConflict`.
- [Cilium v1.18.2 — `pkg/loadbalancer/writer/writer.go`](https://github.com/cilium/cilium/blob/v1.18.2/pkg/loadbalancer/writer/writer.go) — `validateFrontends`, `UpsertServiceAndFrontends` (reject-before-insert), `DeleteServiceAndFrontends`.
- [Cilium v1.18.2 — `pkg/loadbalancer/reflectors/k8s.go`](https://github.com/cilium/cilium/blob/v1.18.2/pkg/loadbalancer/reflectors/k8s.go) — `processServiceEvent` (Upsert/Delete), `reflectorHealth` (no re-queue), `stream.Buffer` coalescing.
- [Cilium v1.18.2 — `pkg/container/insert_ordered_map.go`](https://github.com/cilium/cilium/blob/v1.18.2/pkg/container/insert_ordered_map.go) — first-insertion ordering ("An update will not affect the ordering").
- [Cilium v1.18.2 — `pkg/annotation/k8s.go`](https://github.com/cilium/cilium/blob/v1.18.2/pkg/annotation/k8s.go) — `LBIPAMSharingKey`.
- [Cilium v1.18.2 — `operator/pkg/lbipam/lbipam.go`](https://github.com/cilium/cilium/blob/v1.18.2/operator/pkg/lbipam/lbipam.go) — sharing-group allocation.
- [Cilium v1.18.2 — `pkg/loadbalancer/frontend.go`](https://github.com/cilium/cilium/blob/v1.18.2/pkg/loadbalancer/frontend.go) — `Frontend.Status` / `reconciler.Status` (ID valid only when done).
- [Cilium — LoadBalancer IPAM](https://docs.cilium.io/en/stable/network/lb-ipam/) — "Sharing Keys" (conflicting ports get different IPs within a key).

## Community discussion today

The selected question continues a thread from an opt-in archive: on a shared VIP, a second service claiming an already-owned frontend was rejected with "frontend already owned by another service", and the community's stated understanding was that deleting the owning service does not re-queue the claimant while an annotation-only update on the claimant recovers the state in roughly seven seconds with no BPF flush, with the suspected mechanism being Kubernetes-informer-side event coalescing. The thread named the public files `writer.go`, `k8s.go`, and `insert_ordered_map.go`. This answer confirms that suspected mechanism against public v1.18.2 sources: reject-before-write, no re-queue on owner delete, and first-arrival coalescing ordering.

Other threads this day: a GnuTLS HPACK per-connection decoder thread (published yesterday's question), a Hubble rule-misattribution thread (tracked as an upstream reply), and a ring-0 socket-drop versus context-switch benchmark (too thin to publish).

Channel coverage for this run: the two opt-in archives provided four messages, all covered above. The visible-browser-only sources (Discord, the eunomia-bpf and sched-ext communities, the bpf mailing list, and r/eBPF) could not be reviewed in this run — no visible-browser session was available — so they are marked uncovered, not quiet.
