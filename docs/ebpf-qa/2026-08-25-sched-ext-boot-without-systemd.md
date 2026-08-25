# How can you start a `sched_ext` scheduler at boot without systemd?

**Short answer:** treat the scheduler as a supervised foreground process, not as persistent kernel configuration. A `sched_ext` policy is active only while its BPF scheduler is loaded and running. On OpenRC, runit, dinit, or another init system, create a native service that starts the exact scheduler binary and arguments after the kernel exposes `/sys/kernel/sched_ext`, keeps the process in the foreground, restarts it with bounded delay, and enables that service in the normal boot target. Then verify `/sys/kernel/sched_ext/state` and `/sys/kernel/sched_ext/root/ops`; a green service status alone is not proof that the intended policy is active.

If the scheduler exits, stalls, or is rejected, the kernel disables it and returns tasks to the fair scheduler. That fallback is an important safety property, but it also means that a machine can boot successfully while silently running a different scheduling policy.

## What must survive a reboot

The kernel does not remember a previous `sched_ext` selection as a boot preference. Linux documents `sched_ext` as dynamically enabled when a BPF scheduler is loaded and disabled when it unloads or fails. Therefore the persistent object is an init-system declaration containing:

- the absolute path to a scheduler executable;
- the scheduler's exact, version-matched command-line arguments;
- the point in boot at which the service may start;
- process supervision and restart policy;
- where standard output and error are recorded; and
- the boot target or service bundle that activates it.

The upstream `scx` repository's systemd unit is useful as a statement of this contract even on a non-systemd host. It checks that `/sys/kernel/sched_ext` exists, loads configuration from `/etc/default/scx`, starts the selected scheduler as the service's foreground process, restarts on failure, and joins the multi-user boot target. Translate those semantics into the host's native init format; copying the systemd unit itself does nothing on OpenRC, runit, or dinit.

Before automating startup, run the chosen scheduler once in a controlled session and check its current `--help`. The `sched_ext` ABI is explicitly unstable, and scheduler options also evolve. Boot supervision is the wrong place to discover that a kernel update, package update, binary rename, or stale flag made the command invalid.

## Prefer the direct scheduler process for a simple deployment

For one fixed policy, directly supervising a scheduler such as `scx_lavd` is the smallest design. Keep it in the foreground and let the init system own its PID and lifetime. Do not add `&`, `nohup`, an extra shell daemonizer, or a background mode unless the init system specifically requires one: a supervisor cannot reliably restart a child that immediately detaches from it.

The following are **illustrative service shapes**, not distribution packages. Paths, scheduler choice, arguments, dependencies, logging, and service directories must match the installed system.

### OpenRC

An `/etc/init.d/scx-scheduler` script can use OpenRC's own supervisor:

```sh
#!/sbin/openrc-run

description="Local sched_ext scheduler"
command="/usr/bin/scx_lavd"
command_args="--some-verified-option"
supervisor="supervise-daemon"
respawn_delay=5
respawn_max=5
respawn_period=60

depend() {
    after modules
}
```

OpenRC's guide requires a supervised daemon to remain in the foreground. Its `respawn_delay`, `respawn_max`, and `respawn_period` controls can stop an invalid scheduler from becoming an unlimited hot restart loop. The dependency name is distribution-specific; `after modules` is only an example. A stronger local pre-start check should refuse to launch if `/sys/kernel/sched_ext` is absent. Enable the service through the distribution's normal OpenRC runlevel mechanism only after a manual start succeeds.

### runit

In a runit service directory, `run` can replace itself with the scheduler:

```sh
#!/bin/sh
exec 2>&1
exec /usr/bin/scx_lavd --some-verified-option
```

`runsv` starts `./run` and normally restarts it after exit. Make the service directory part of the supervised boot set using the distribution's normal link or package mechanism. Because runit intentionally restarts services, check the log after the first boot and do not let a permanently incompatible binary cycle unnoticed. A small, deliberate delay in a local `finish` policy may be appropriate when the distribution does not already provide sufficient backoff.

### dinit

A process service can express the same lifecycle:

```text
type = process
command = /usr/bin/scx_lavd --some-verified-option
restart = true
waits-for: local-preconditions
```

Dinit's documented process service expects a foreground command and supports automatic restart. Place the service description in the configured service directory, replace `local-preconditions` with real services on that system, and add the scheduler to the appropriate boot dependency chain. Do not copy a dependency name from another distribution merely because it sounds plausible.

In every case, keep mutable arguments in a root-owned configuration file, use absolute paths, and avoid evaluating untrusted text through a shell. Start as root unless the selected scheduler and local policy have a deliberately tested, narrower privilege model.

## `scx_loader` is a different deployment model

`scx_loader` is a management daemon that exposes scheduler control over the system D-Bus; `scxctl` is its command-line client. Its configuration can select a `default_sched` and mode, so starting the loader can in turn start the configured scheduler. This is useful when policies must be switched through one control plane.

It also introduces more boot prerequisites. The upstream loader currently ships a systemd D-Bus service and systemd hardening unit. A non-systemd port therefore needs all of the following, not merely a call to `scxctl`:

1. a running system D-Bus;
2. the loader's D-Bus service and policy/interface files installed in the locations used by that system;
3. a native init service supervising `scx_loader` in the foreground;
4. an explicit dependency on the system bus and kernel prerequisites; and
5. a verified `default_sched` configuration or a separate, ordered control action.

If those pieces are not needed, direct supervision avoids an extra daemon and authorization surface. If they are needed, manage the loader as the durable service and treat the scheduler process it owns as subordinate state. Do not supervise both independently; two controllers can race to replace the active `sched_ext` policy.

## Verify kernel state after every boot

The kernel provides a direct status surface:

```sh
test -d /sys/kernel/sched_ext
cat /sys/kernel/sched_ext/state
cat /sys/kernel/sched_ext/root/ops
cat /sys/kernel/sched_ext/enable_seq
```

After the service starts:

- `state` should report `enabled`;
- `root/ops` should identify the expected scheduler operations;
- `enable_seq` should show that a scheduler has been enabled; and
- the supervised process should still be alive without a restart loop.

These observations distinguish several failures that a generic “service started” check cannot:

- **No `/sys/kernel/sched_ext`:** the running kernel lacks the facility or the required kernel configuration.
- **Service inactive:** boot activation, dependency ordering, executable path, or permissions are wrong.
- **Service repeatedly exits and state is disabled:** capture the first verifier or loader error; stop the loop and test one invocation manually.
- **State is enabled but `root/ops` is unexpected:** another controller or package selected a different scheduler.
- **The scheduler worked before a kernel upgrade but no longer loads:** test the exact kernel, BTF, scheduler build, and command-line combination; do not disguise an ABI mismatch with infinite restart.

Exercise the failure path before relying on it. Stop the scheduler deliberately and confirm that tasks continue under the kernel's fair class, the supervisor records the failure, restart behavior matches policy, and the state files return to the expected values. Keep a documented way to disable the boot service and remain on the fair scheduler. That rollback is more valuable than trying to make the boot path incapable of failure.

## References

- [Linux kernel `sched_ext` documentation: dynamic enablement, failure fallback, status files, and ABI warning](https://docs.kernel.org/scheduler/sched-ext.html)
- [`scx` upstream systemd service](https://github.com/sched-ext/scx/blob/main/services/scx.service)
- [`scx` upstream default service configuration](https://github.com/sched-ext/scx/blob/main/services/scx)
- [`scx` service documentation](https://github.com/sched-ext/scx/blob/main/services/README.md)
- [`scx_loader` architecture and system D-Bus interface](https://github.com/sched-ext/scx-loader/blob/main/README.md)
- [`scx_loader` configuration, including `default_sched`](https://github.com/sched-ext/scx-loader/blob/main/crates/scx_loader/configuration.md)
- [`scx_loader` upstream systemd unit](https://github.com/sched-ext/scx-loader/blob/main/services/scx_loader.service)
- [`scxctl` client documentation](https://github.com/sched-ext/scx-loader/blob/main/crates/scxctl/README.md)
- [OpenRC service-script guide](https://github.com/OpenRC/openrc/blob/master/service-script-guide.md)
- [OpenRC `supervise-daemon` guide](https://github.com/OpenRC/openrc/blob/master/supervise-daemon-guide.md)
- [runit `runsv(8)` manual](https://smarden.org/runit/runsv.8)
- [Dinit service-description overview](https://github.com/davmac314/dinit/blob/master/README.md)
- [bpftime fixed-hash iteration fix and regression test](https://github.com/eunomia-bpf/bpftime/pull/658)
- [OpenTelemetry eBPF metric-label change](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/pull/3168)
- [OpenTelemetry GenAI evidence-reference proposal](https://github.com/open-telemetry/semantic-conventions-genai/issues/470)
- [Linux BPF patch discussion: cross-page USDT instruction patching](https://lore.kernel.org/bpf/20260825150444.31603-1-jiayuan.chen@linux.dev/T/#t)
- [Linux BPF patch discussion: page ownership in an XDP path](https://lore.kernel.org/bpf/20260824030257.263179-1-jiayuan.chen@linux.dev/T/#t)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected question came from the 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, workspace and channel identities, message links, exact times, private topology, raw logs, and searchable wording have been removed. No raw transcript was retained, and no social interaction was performed.

### Boot automation needs a kernel-state acceptance test

The strongest unresolved question was how to preserve a `sched_ext` choice across reboot on a machine that does not use systemd. The important reframing was that the policy itself is not persisted: boot automation must recreate a live attachment by supervising a userspace scheduler. The upstream [kernel status files](https://docs.kernel.org/scheduler/sched-ext.html) make the acceptance test independent of whether the local init system is OpenRC, runit, dinit, or something else.

This led to two practical concerns. First, a service manager should supervise the foreground process and bound retries, because kernel fallback keeps the host usable even when an invalid command exits immediately. Second, service liveness and scheduler activation are separate facts. A wrapper or management daemon can stay alive while no intended scheduling operations are attached, so `state` and `root/ops` belong in post-boot checks.

### Boundary conditions dominated runtime and kernel work

A userspace BPF runtime fix addressed iteration in a fixed hash map whose allocated bucket count had been adjusted to a prime number. Lookups could still reach entries, but key iteration used the unadjusted size and could skip buckets. The public [fix](https://github.com/eunomia-bpf/bpftime/pull/658) makes lookup and iteration share the effective capacity and adds a regression test. The lesson is broader than that map: APIs that traverse a structure must use the same realized geometry as APIs that populate it.

Kernel discussion repeatedly returned to ownership and boundary failures. One patch handled a userspace-static-tracing instruction sequence that crosses a page boundary; another examined page ownership and release in an XDP path. Memory reclaim under a cgroup limit, program signing, stream-buffer sizing, private BPF stacks, verifier bounds, and HID BPF changes were also active. Across those topics, the most valuable tests force the unusual boundary or failure path instead of proving only the common case.

### Telemetry changes need migration semantics, not only cleaner schemas

An observability implementation proposed removing service identity from default metric labels while keeping it as resource data and allowing explicit opt-in. That avoids duplicating identity on every data point, but it is still a breaking query surface. Users need a migration path—such as resource-to-target metadata joins or a temporary compatibility option—rather than discovering the new label set from failed dashboards. The public [change](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/pull/3168) remains the source of truth for its current status.

A previous discussion about attaching verifiable evidence to GenAI evaluation results also advanced into a public [specification issue](https://github.com/open-telemetry/semantic-conventions-genai/issues/470). That follow-up was not promoted into a second daily question. The useful distinction is unchanged: a standard can define a reference, digest, and media type without claiming that referenced content is trustworthy or turning telemetry storage into an artifact archive.

### Quiet targets were still checked

Several project help surfaces had no substantive message in the daily window, and one public practitioner forum's newest post was more than a week old. A networking community was active mainly around contributor onboarding, documentation, and meeting logistics. Other project-specific surfaces contained automated build notices, introductions, older threads, or no messages. Those targets were recorded as accessible and quiet, not converted into zero activity or used as fallback evidence.
