# 在 Cilium LoadBalancer 中，拥有共享 VIP 端口的冲突 Service 被删除后，为什么前端会一直缺失，控制器应如何恢复它？

前端没有坏掉——它从未被安装过。Cilium 的 kube-proxy 替代方案对服务的 frontends 做原子校验，当第二个服务去认领另一个服务已拥有的 (VIP, 端口, 协议) 时，整个 upsert 被拒绝：`ErrFrontendConflict` 在 `validateFrontends` 里先于服务行插入触发，于是属主服务保留它仍在工作的 frontends，而认领者新加的端口从未出现在 BPF map 里。删除属主服务时，reflector 的 delete 处理器只移除属主服务的 frontends 和服务行；它没有把被拒绝的认领者重新排队或重新处理的路径，upsert 错误只被报告给 reflector 的降级健康状态、从不会被重试。认领者只有在针对它的新 upsert 事件到达时才会恢复（例如一次产生非空 patch 的 Kubernetes 更新）。这是一个 reconcile 缺口，不是文档化的行为。支持的共享 VIP 端口迁移方式是按顺序交接——先释放源服务的端口、再让目标认领，或保证有一次后续的认领者事件——并把 `Frontend.Status` 达到 `done`（而不是期望状态被接受）当作数据面收敛信号。

## 机制

Cilium 的 kube-proxy 替代方案把每个 LoadBalancer 服务的 frontends 放进 BPF 支撑的状态表里，从 Kubernetes 服务和端点事件做 reconcile。写路径在 `pkg/loadbalancer/writer/writer.go`：

- `UpsertServiceAndFrontends(txn, svc, fes...)` 先对完整的前端集合运行 `validateFrontends(txn, fes...)`，之后才用 `w.svcs.Insert(txn, svc)` 碰服务表。
- `validateFrontends` 逐个检查前端地址；若该地址已被另一个服务拥有，就返回 `ErrFrontendConflict`，包装成 `frontend already owned by another service: <vip:port/proto> is owned by <ns/name>`。
- 出错时整个 upsert 立即返回：服务行不插入、新 frontend 不写、后端不释放。属主服务已有的 frontends 原样留在 BPF map 里。

这就是"前端持续缺失"的状态：认领者服务行保留旧的端口集合（其幸存 frontends 继续工作），冲突端口从未被创建。

读侧在 `pkg/loadbalancer/reflectors/k8s.go`。它的事件处理器有两个分支：

- `resource.Upsert` 把 Service 转成 frontends 加 backends，调用 `UpsertServiceAndFrontends`；任何错误都经 reflector 的健康对象（`rh.update` → 降级状态加告警日志）记录。没有重试。
- `resource.Delete` 调用 `DeleteServiceAndFrontends(txn, name)`，移除被删服务的 frontends 然后移除其服务行。它不对其他服务重新评估被释放的地址，也不重新排队某个此前 upsert 被拒的认领者。

因此恢复需要针对认领者的一个新 upsert 事件。产生非空 patch 的 Kubernetes Update（最小例子是改一个 annotation）会生成这个事件；空 patch 的 apply 不会。

事件合并解释了顺序失败。`k8s.go` 用 `stream.Buffer` 以 500 事件 / 500ms 窗口把服务和端点事件合并进 `InsertOrderedMap`。该容器的 `InsertOrderedMap.Insert` 文档写明"更新不影响顺序"，所以一个批次内事件按先到顺序处理。若认领事件先于释放事件到达，认领先被处理（被拒、且不再重试），随后释放才腾空前端，而没有任何机制重跑认领。释放与认领可以落在同一个合并批次里，所以"属主已删除"并不蕴含"认领在释放之后被重处理过"。

`Frontend.ID`（BPF service map 的 key）在 `pkg/loadbalancer/frontend.go` 里被文档为仅当 `Frontend.Status` 为 `done` 时有效，`cilium service list` 按 frontend 暴露 `Status`、`Since`、`Error`。这就是受支持的数据面收敛信号。

## 验证与调试路径

对着公开源码复现并确认"先拒后写"的序列：

1. 两个服务通过 `lbipam.cilium.io/sharing-key` 共用同一 VIP，给第二个服务加一个第一个服务拥有的端口。认领者的 upsert 打出 `frontend already owned by another service: <vip:port> is owned by <ns/name>`（`ErrFrontendConflict`）；`cilium service list` 里看不到认领者的 frontend，而属主已有 frontends 继续服务。
2. 删除属主服务。`DeleteServiceAndFrontends` 对属主执行；被腾空的前端不会自动出现在认领者名下，因为被拒的认领者没有被重新排队。
3. 给认领者生成一次后续 upsert（非空 patch 的 Update）。前端被安装；`Frontend.Status` 从 pending 转到 done，ID 变为有效。
4. 观察合并边界：把认领与对应释放放进同一个约 500ms 窗口，确认认领先于释放处理（先到顺序），且先到的认领会让前端缺失，直到下一个认领者事件。
5. 用 `Frontend.Status` = done（经 `cilium service list`）判断收敛，而不是 Kubernetes Service 对象被接受：期望状态被接受不等于数据面收敛。

## 局限

缺口在于 reconcile 器没有所有权交接：删除属主服务不会把被拒的认领者重新排队。文档没有把它作为受支持的迁移路径，也没有找到已发布的修复——当前公开 Cilium 版本里 `ErrFrontendConflict` 仍在。受支持的做法是排序而非重试：先释放源服务的端口再让目标认领，或保证一次后续认领者事件（仅改 annotation 的 Kubernetes 更新是最小的非空 patch 触发器，但这是顺带的实现行为，不是文档化 API）。不要同时让两个服务占用同一个 (VIP, 端口)。

硬边界是源删除在 K8s API 侧的完成：它不保证 informer 在认领者 upsert 之前就交付并处理了释放事件，而 500 事件/500ms 的合并窗口可以吸收一次释放并按乱序重放。没有任何单次事件检查能证明数据面安全；唯一诚实的收敛信号是 `Frontend.Status` 达到 done。

## 参考

- [Cilium v1.18.2 — `pkg/loadbalancer/errors.go`](https://github.com/cilium/cilium/blob/v1.18.2/pkg/loadbalancer/errors.go) — `ErrFrontendConflict`。
- [Cilium v1.18.2 — `pkg/loadbalancer/writer/writer.go`](https://github.com/cilium/cilium/blob/v1.18.2/pkg/loadbalancer/writer/writer.go) — `validateFrontends`、`UpsertServiceAndFrontends`（先拒后插）、`DeleteServiceAndFrontends`。
- [Cilium v1.18.2 — `pkg/loadbalancer/reflectors/k8s.go`](https://github.com/cilium/cilium/blob/v1.18.2/pkg/loadbalancer/reflectors/k8s.go) — `processServiceEvent`（Upsert/Delete）、`reflectorHealth`（不重新排队）、`stream.Buffer` 合并。
- [Cilium v1.18.2 — `pkg/container/insert_ordered_map.go`](https://github.com/cilium/cilium/blob/v1.18.2/pkg/container/insert_ordered_map.go) — 首次插入顺序（"更新不影响顺序"）。
- [Cilium v1.18.2 — `pkg/annotation/k8s.go`](https://github.com/cilium/cilium/blob/v1.18.2/pkg/annotation/k8s.go) — `LBIPAMSharingKey`。
- [Cilium v1.18.2 — `operator/pkg/lbipam/lbipam.go`](https://github.com/cilium/cilium/blob/v1.18.2/operator/pkg/lbipam/lbipam.go) — 共享组分配。
- [Cilium v1.18.2 — `pkg/loadbalancer/frontend.go`](https://github.com/cilium/cilium/blob/v1.18.2/pkg/loadbalancer/frontend.go) — `Frontend.Status` / `reconciler.Status`（done 时 ID 才有效）。
- [Cilium — LoadBalancer IPAM](https://docs.cilium.io/en/stable/network/lb-ipam/) — "Sharing Keys"（冲突端口在同一 key 下分配到不同 IP）。

## 当日社区讨论

选取的问题延续自 opt-in 存档的一个讨论：共享 VIP 上，第二个服务认领已被占用的 frontend 被拒，报"frontend already owned by another service"；社区陈述的理解是删除属主服务并不会把认领者重新排队，而对认领者做一次仅改 annotation 的更新约七秒内恢复状态、无 BPF flush，怀疑机制在 Kubernetes informer 侧的事件合并。讨论点名的公开文件是 `writer.go`、`k8s.go` 与 `insert_ordered_map.go`。本篇对照 v1.18.2 公开源码确认了该怀疑机制：先拒后写、属主删除不重新排队、合并按先到顺序处理。

当日其他讨论：一条 GnuTLS HPACK 每连接解码器讨论（即昨日发布的问题）、一条 Hubble 规则归因讨论（作为上游回复跟踪）、一条 ring-0 socket drop 与 context switch 的基准对比（信号太薄未发布）。

本次运行的渠道覆盖：两个 opt-in 存档共四条消息，全部如上覆盖。visible-browser-only 来源（Discord、eunomia-bpf 与 sched-ext 社区、bpf 邮件列表、r/eBPF）本次未能审阅——没有可用的 visible-browser 会话——因此标记为未覆盖，而非平静。
