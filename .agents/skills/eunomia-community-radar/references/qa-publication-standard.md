# Public eBPF Q&A Standard

Use this reference only after a monitored question passes the editorial gate.

## Reader Outcome

The page should let an engineer understand the answer without knowing the
original conversation. It should explain the decisive boundary, show how to
verify it, and state where the answer stops applying.

## Recommended Shape

1. Use the real practitioner question as the title after anonymization.
2. Start with a direct short answer.
3. Explain the mechanism that makes the answer true.
4. Give a small verification or debugging path.
5. State the important limitation or alternative.
6. Link the public primary sources used to verify the answer.

Use only the sections the question needs. This is a useful shape, not a required
template.

## Anonymization Test

Before publishing, confirm that the page contains none of the following:

- a person's name, handle, avatar, employer, or team;
- a workspace, server, channel, or message URL;
- an exact timestamp or sequence that identifies the source thread;
- private logs, hostnames, IP addresses, repository names, credentials, or
  deployment topology;
- copied wording that can be searched back to the participant; or
- an assertion that a community member, customer, or organization endorsed the
  answer.

If removing those details changes the technical answer, do not publish it.

## Evidence And Privacy

Prefer kernel documentation, project documentation, upstream source code,
standards, and papers. A community message is a lead, not public evidence.

Do not collect sensitive data merely to demonstrate that it can be detected.
When the subject itself concerns secrets or private payloads, explain which
component can see plaintext, what leaves that component, and how to verify that
raw values are absent from maps, buffers, logs, traces, and exported telemetry.

## Publication Floor

Daily monitoring does not create a daily content quota. Publish only when the
answer is technically useful, independently verifiable, non-duplicative, and
safe after anonymization. Publish at most one Q&A per day.
