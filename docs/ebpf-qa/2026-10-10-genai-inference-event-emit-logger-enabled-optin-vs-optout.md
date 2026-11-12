# Why should a GenAI instrumentation gate inference events on the log pipeline severity instead of a dedicated emit flag, and is logger enabled an opt-in or an opt-out control?

The gate belongs in the standard log pipeline, not in a dedicated environment-variable flag. The GenAI inference event is specified at DEBUG severity with a Recommended requirement level, so a pipeline that keeps ordinary INFO logs already drops it; the instrumentation then asks `Logger.enabled` with that severity before building the record, which skips both the allocation and the capture of message content when the event will not survive. The decisive boundary is what `Logger.enabled` is *not*: with the SDK default logger and no configured minimum severity it returns true for everything, so by itself it is not an opt-in. It becomes an opt-out control only when the pipeline enforces a floor - a minimum severity, a trace-sample decision, or an event-name disable - and the control is then declarative, not a per-feature flag.

## The mechanism

The OpenTelemetry GenAI conventions define the client-inference event as Recommended and require instrumentations to set its severity to DEBUG (severity number 5) when recording it. DEBUG sits below the INFO floor that most log pipelines use, so the event is opt-out by default severity: an ordinary deployment never receives it unless it deliberately keeps DEBUG-level logs. The conventions are explicit that the event exists to store input and output details independently of traces, which is exactly why the cost of constructing it - and the privacy cost of touching message content - matters.

The Python GenAI utility package is moving the emission decision onto that pipeline. Its `_maybe_create_event` path now checks whether the event is enabled through the logger at DEBUG severity and only builds and emits the log record when the check passes, with a graceful fallback for loggers whose `enabled` method does not yet accept a severity argument. The dedicated `EMIT_EVENT` style environment flag is being removed in favor of the content-capture flag plus this logger check: the flag decides whether prompt and response content is attached, and the pipeline decides whether the event itself reaches anyone.

The enabling piece on the SDK side is declarative configuration: minimum-severity and trace-based filtering are wired through `Logger.enabled`, and declarative config can disable events by instrumentation name, and will additionally let you disable them by minimum severity or by the trace sample flag. That is the boundary in one sentence - the feature flag answers "should I look at content at all", the pipeline answers "will this event survive", and the two are separated so that a DEBUG-level diagnostic event cannot silently ride along with every INFO log.

## Verification and debugging path

1. Open the GenAI client-inference conventions, Events section: the client-inference event is Recommended and says instrumentations SHOULD set the severity to DEBUG (severity number 5). That single line is the reason the event is quiet under an INFO pipeline.
2. Read the opentelemetry-python GenAI utility change that decouples event emission: confirm the emit path now consults `logger.enabled` at the event severity, skips record creation when it returns false, and that the unconditional environment-flag mutation inside the instrumentation was removed.
3. Read the SDK-side change that adds minimum-severity and trace-based filtering to `Logger.enabled` and wires it through declarative configuration: that is what turns `enabled` from "defaults to true" into a real gate.
4. In a local app, build a log pipeline with an INFO floor and confirm `enabled` returns false for the DEBUG event, so the record is never constructed; lower the floor or disable the event by name in declarative config and confirm the transition. Verify with the content-capture flag off that prompt and response bodies are absent from the event payload even when the event is emitted.

## The limitation

- With no minimum severity configured, the SDK logger's `enabled` returns true for every severity, so `Logger.enabled` is not a standalone opt-in today; the opt-in boundary is "does your pipeline keep DEBUG events", and the answer is usually no unless you set it up.
- The severity-aware `enabled` call needs a current opentelemetry API release; older or custom loggers that do not accept the severity argument must be handled by a fallback, or the check silently degrades.
- The event is in Development status and not yet available in every language, so the same pattern has uneven support across SDKs.
- The event carries input and output details by design; the content-capture flag controls attachment, but the severity gate is what keeps the event itself out of a normal pipeline.

## References

- [OpenTelemetry semantic-conventions-genai — client inference events](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md) — the client-inference operation-details event: Recommended requirement level, SHOULD carry DEBUG severity (number 5), and may store request and response details independently of traces.
- [opentelemetry-python pull request 5724](https://github.com/open-telemetry/opentelemetry-python/pull/5724) — minimum-severity and trace-based filtering for `Logger.enabled`, wired through declarative configuration, plus instrumentation-name event disabling.
- [opentelemetry-python-genai pull request 836](https://github.com/open-telemetry/opentelemetry-python-genai/pull/836) — the GenAI utility package decoupling event emission: the emit decision moves to the logger check at DEBUG severity and the dedicated emit flag is removed in favor of the content-capture flag.
- [OpenTelemetry Python logs API](https://opentelemetry.io/docs/languages/python/logs/) — the Python Logs API surface, including `LoggerProvider`, `Logger`, and `enabled`.

## Community discussion today

Two opt-in archive channels (the two CNCF OpenTelemetry instrumentation channels) produced twelve messages this run. The strongest new thread is the one this page answers: a pull request in the Go agent-development toolkit was pointed at the GenAI utility change and asked whether it should follow the same logger-check pattern, and the open question is whether the inference event should stay opt-in or move to opt-out. The answer in the thread is the boundary above: the logger check defaults to enabled, so it is not an opt-in mechanism on its own, and the opt-out route is the declarative pipeline - disable by event name, by minimum severity, or by the trace sample flag. A second thread asks whether a Node.js GenAI instrumentation repository is planned to match the Python one; the reply is that instrumentation is landing in the JavaScript contrib repository and the auto-generated conventions in the core JavaScript SDK, but no public decision yet on whether a dedicated repository is wanted. Two further threads continue topics already published: the dedicated skill-definitions attribute for agent traces (the 10-05 entry) and the Kubernetes model-serving-signals initiative (the 10-09 entry).

Channel coverage for this run: the two opt-in archives supplied twelve messages. The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.
