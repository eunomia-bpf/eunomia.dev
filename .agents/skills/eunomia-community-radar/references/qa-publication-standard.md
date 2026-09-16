# Public eBPF Q&A Standard

Use this reference when drafting and reviewing the daily public Q&A.

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

The H1 must read as a plain technical question. The publisher renders each
question as the page H1, and the publication validator re-fetches that H1 from
the live page and requires the exact Markdown title string to appear verbatim.
Because the site's H1 pipeline strips underscores from a plain-text title (so
`bpf_probe_read_user` renders as `bprobereaduser`) and HTML-encodes apostrophes,
keep the H1 free of underscores, backticks, and apostrophes; put helper names
inside the H1 as ordinary words or move them into the answer body where
backticks belong. An H1 that fails the live-content check forces an extra
redeploy just to fix the title.

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

Every daily run publishes one technically useful, independently verifiable,
non-duplicative Q&A that remains safe after anonymization. When no single
message is sufficient, combine related same-day signals, use a real unresolved
question from the monitored communities regardless of age, or select a genuine
question from permitted public community pages and public primary
documentation. Report access and coverage gaps honestly in the run notes; a
coverage gap is always reported and never blocks publication. Never invent a
question, fabricate evidence, or publish a placeholder.

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
