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
6. Add a References section with the public primary sources used.
7. End with an anonymized analysis of the day's wider community discussion.

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

Every successful daily run publishes one technically useful, independently
verifiable, non-duplicative Q&A that remains safe after anonymization. When no
single message is sufficient, combine related same-day signals or use the most
recent unresolved recurring question from the previous seven days. Never invent
a question or publish a placeholder. If access or evidence is insufficient, the
run fails explicitly instead of claiming completion.

## Daily Community Briefing

The Q&A page also carries the daily community report. After the references:

- state how many communities and allowlisted channels were actually reviewed;
- group discussion by technical theme rather than source or participant;
- explain the concrete question or symptom, likely mechanism, practical next
  diagnostic or resolution step, and any unresolved boundary for each
  substantive theme;
- cite the public primary references used for these concise technical answers;
- preserve technical depth while removing identity and deployment details;
- distinguish quiet channels from inaccessible channels; and
- omit names, handles, employers, channel names, message URLs, exact timestamps,
  raw quotes, and distinctive deployment details.

When several substantive discussions are present, give them enough space to be
useful on their own. A one-sentence trend label is not a community report; the
discussion section should normally be at least twice as detailed as such a
compressed summary.
