# agentpprof: pprof-style profiles for AI coding-agent sessions

An AI coding agent can finish a task and leave behind hundreds of events: user
prompts, model calls, shell commands, file edits, package downloads, test runs,
and retries. Traditional logs can answer "what happened next?" if you read them
line by line. They are much weaker at answering the question developers usually
ask after a long run: where did the agent spend its work, what repeated, and
which prompts caused the expensive or risky system effects?

`agentpprof` is a profiling tool for that question. It reads local Codex and
Claude Code session history through the shared `agent-session` parser, projects
the session into semantic stacks, and writes outputs that existing profiling
tools already understand: Go pprof protobuf, folded stacks, SVG flamegraphs, and
redacted JSON.

The profiles are not CPU profiles. A wide frame does not mean the agent used
more CPU. Width means the chosen view's weight: activity count, system-effect
count, file-effect count, network-effect count, or model token count. The goal
is to make an agent session browsable like a performance profile, without
pretending that every question has the same metric.

## Why a profiler instead of another trace view?

Trace views are good when you already know where to look. If a single command
failed at 14:03, a timeline can show the surrounding tool calls and model
messages. Long agent sessions have a different failure mode: there is too much
trace. The user wants to know whether the agent kept re-reading the same files,
spent most of its model budget on review, retried tests in a loop, or contacted
external services during one prompt.

A flamegraph is useful here because it compresses repeated work. The stack says
which context the event belongs to, and the width says how much weight that
context accumulated. When many prompts share the same pattern, they merge. When
one prompt creates a distinct system effect, it remains visible as a branch.

`agentpprof` keeps this model explicit. It does not try to produce one universal
"agent flamegraph". Instead, it exposes several projections over the same
session:

| View | Width means | Primary question |
| --- | ---: | --- |
| `tokens` | reported token count (input/output/cache) | Which prompts consumed the most model budget? |
| `time` | duration in seconds | How long did each prompt/activity take? |
| `files` | file/path effect count | Which prompts touched which parts of the repository? |
| `network` | network/domain effect count | Which prompts contacted which domains? |

Start with `tokens` to find cost hotspots. Use `time` to see where wall-clock
time went. Use `files` and `network` for security audits.

## Install

After release, install from crates.io:

```bash
cargo install agentpprof
```

You can also download the `agentpprof` binary from the AgentSight GitHub
release artifacts. The AgentSight release pipeline builds and smoke-tests both
`agentsight` and `agentpprof` from the same release tag.

From a source checkout:

```bash
cargo run --manifest-path agentpprof/Cargo.toml -- --version
cargo run --manifest-path agentpprof/Cargo.toml -- -o agent.pb.gz
```

## First profile

Generate a token profile for the current repository:

```bash
agentpprof --project-root . --view tokens -o tokens.pb.gz
```

Open the pprof profile with standard Go tooling:

```bash
go tool pprof -top tokens.pb.gz
go tool pprof -http=:0 tokens.pb.gz
```

Generate a browser-openable flamegraph instead:

```bash
agentpprof --project-root . --view tokens -o tokens.svg
```

The extension chooses the output format when `--format` is not provided:

```bash
agentpprof -o tokens.pb.gz  --view tokens   # pprof protobuf, gzip-compressed
agentpprof -o time.folded   --view time     # folded stack text
agentpprof -o files.svg     --view files    # standalone SVG flamegraph
agentpprof -o network.json  --view network  # redacted JSON summary and stacks
```

## Example Flamegraphs

The examples below were generated from 2533 real local bpf-benchmark development
sessions (Codex + Claude Code). They demonstrate what insights each view provides.

### Tokens View

**Question:** Which activities consumed the most model budget?

![Tokens flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/bpf-benchmark-tokens.svg)

The token distribution reveals that paper writing (`prompt:paper`, 124 frames) dominated the model budget, consuming approximately 20% of total tokens across all sessions. Iterative editing (`prompt:edit`, 92 frames) and code review (`prompt:review`, 80 frames) follow as the next largest categories, indicating that refinement activities—not initial generation—drove most of the cost. At the session level, review-focused sessions (`session:review`, 251 frames) span the widest bar, suggesting that careful inspection workflows are more token-intensive than benchmarking or naming tasks. The presence of `prompt:unmatched` (100 frames, ~14%) indicates room for additional tagging rules to improve semantic coverage.

### Time View

**Question:** Where did wall-clock time go across 2533 sessions?

![Time flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/bpf-benchmark-time.svg)

Wall-clock time largely mirrors token consumption, with paper writing (`prompt:paper`, 186 frames) taking the most time. However, review prompts (`prompt:review`, 130 frames) appear proportionally wider in the time view than in tokens, suggesting these prompts involve longer response latencies or more deliberation. Continuation prompts (`prompt:continue`, 66 frames) appear frequently throughout sessions, reflecting a workflow pattern where complex tasks required multiple follow-up exchanges. The overall similarity between time and token distributions indicates no single activity category suffers from unusually slow model responses or system bottlenecks.

### Files View

**Question:** Which parts of the codebase were touched and how?

![Files flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/bpf-benchmark-files.svg)

File access patterns show that nearly all activity remained within project boundaries (paths prefixed with `repo/`), with no unexpected access to external directories. The `docs/` subtree appears frequently, consistent with the paper-writing activity identified in the tokens view. The flamegraph distinguishes between read and write effects, revealing the balance of inspection versus modification. External paths (`external/home`, `external/tmp`) are minimal, confirming that the agents operated within expected filesystem boundaries—a useful signal for security audits.

### Network View

**Question:** Which external services were contacted?

![Network flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/bpf-benchmark-network.svg)

Network activity is sparse relative to file operations, confirming that most development work occurred locally without external dependencies. The contacted domains—`github.com` for version control and `api.anthropic.com` for model inference—are expected and benign. No unexpected third-party services appear, which is reassuring from a supply-chain security perspective. Process chains visible in the upper frames show which tools initiated network requests (e.g., `git push`, `curl`), enabling attribution of network activity to specific agent actions

See `docs/flamegraph-example/bpf-benchmark.sh` for the generation script with tag rules

## What data does it read?

`agentpprof` reads agent-native local session history. Today that means Codex
and Claude Code JSONL session files parsed through the `agent-session` crate.
It does not load eBPF probes, require root, or record a live process. It is the
offline profiling side of AgentSight: use `agentsight` to observe live system
behavior, and use `agentpprof` to aggregate already-recorded agent sessions.

By default, it scans recent local sessions that match `--project-root`:

```bash
agentpprof --project-root /path/to/repo --view tokens -o tokens.svg
```

For repeatable analysis, pass explicit session files:

```bash
agentpprof \
  --project-root /path/to/repo \
  --session-file ~/.codex/sessions/.../session.jsonl \
  --session-file ~/.claude/projects/.../session.jsonl \
  --view tokens \
  -o tokens.folded
```

Useful selectors:

```bash
agentpprof -o tokens.svg --agent codex
agentpprof -o tokens.svg --session-id 019ec5
agentpprof -o tokens.svg --session-tag debug
agentpprof -o tokens.svg --prompt-tag review
```

## The stack model

A stack is a projection, not a literal call stack. The lower frames provide
context, and the upper frames describe the activity being counted. The exact
shape depends on the view.

The `tokens` view uses model budget as the width:

```text
project:agentsight;agent:claude;session:profile;prompt:debug;call:llm/debug;model:claude-opus-4-6;kind:input 4200
project:agentsight;agent:claude;session:profile;prompt:debug;call:llm/debug;model:claude-opus-4-6;kind:output 980
project:agentsight;agent:claude;session:profile;prompt:debug;call:llm/debug;model:claude-opus-4-6;kind:cache 150000
```

The `time` view uses wall-clock duration (seconds) as the width:

```text
project:agentsight;agent:claude;session:profile;prompt:debug;kind:llm 45
project:agentsight;agent:claude;session:profile;prompt:debug;kind:tool 12
project:agentsight;agent:claude;session:profile;prompt:debug;kind:prompt 2
```

The `files` view makes repository areas the main branch:

```text
project:agentsight;agent:codex;session:release;prompt:docs;path:docs/flamegraph;effect:write;status:ok 1
```

The `network` view centers domains:

```text
project:agentsight;agent:codex;session:release;prompt:publish;domain:crates.io;process:cargo;status:ok 1
```

The right projection depends on your question: tokens for cost, time for
performance, files for impact, network for security.

## Tagging

The most important frames are `session:*`, `prompt:*`, and `call:llm/*`. Raw
prompts are too long and too private to use as flamegraph labels, so
`agentpprof` maps them to short one-word tags.

The default tagger is deterministic and local:

```bash
agentpprof -o tokens.svg --tagger regex
```

It uses built-in keyword rules and produces stable tags such as `debug`,
`review`, `test`, `docs`, `release`, `profile`, or `design`. This is the safest
default for CI, public artifacts, and reproducible analysis.

Project-specific rules can be layered on top:

```bash
agentpprof -o tokens.svg \
  --tagger regex \
  --tag-rule prompt:review='(?i)review|diff|regression' \
  --tag-rule prompt:test='(?i)cargo test|pytest|unit test' \
  --tag-rule session:release='(?i)release|publish|crates\\.io'
```

Rules use:

```text
KIND:TAG=REGEX
```

`KIND` may be `session`, `prompt`, `llm`, or `all`. `TAG` must be one lowercase
English word between 3 and 12 letters. Rules are evaluated in command-line
order before the built-in rules.

For model-produced tags, run a llama.cpp-compatible server and use the LLM
tagger:

```bash
llama-server -m /path/to/model.gguf --port 8080
agentpprof -o tokens.svg --tagger llm --llama-url http://127.0.0.1:8080
```

LLM tags are cached under the user cache directory by default, for example
`$XDG_CACHE_HOME/agentpprof/tags.json`. Use `--cache` to choose another file,
or `--no-cache` to avoid saving new tags.

The model is not asked to validate whether the agent was correct. It only names
short semantic regions. Correctness and safety still require separate evidence.

## Privacy and redaction

Local agent histories can contain prompts, tool outputs, paths, commands,
repository names, and model responses. `agentpprof` is conservative by default:

- SVG, pprof, and folded outputs contain stack labels and weights, not raw
  prompts or model responses.
- JSON output redacts previews unless `--include-previews` is set.
- Absolute paths outside the selected project root are grouped into stable
  buckets such as `external/home`, `external/tmp`, `external/codex`, and
  `external/claude`.
- Private-looking domains are collapsed instead of exposing user-specific
  hostnames.

Use explicit `--session-file` inputs when you need repeatability. Use
`--include-previews` only for private debugging or already-sanitized sessions.

## Using agentpprof with AgentSight

`agentpprof` and `agentsight` answer related but different questions.

`agentsight` is the live and recorded system observer. It uses eBPF, TLS traffic
capture, process monitoring, and materialized views to show what an agent does
at runtime. Use it when you need live visibility, process trees, file effects,
network destinations, or saved SQLite traces.

`agentpprof` is the semantic profiler for local agent history. Use it when you
want aggregation: token cost hotspots, time spent per prompt, or folded stacks
that can be compared across sessions.

A practical workflow is:

```bash
sudo agentsight record -- claude
agentsight report
agentpprof --project-root . --view tokens -o tokens.svg
agentpprof --project-root . --view time -o time.svg
agentpprof --project-root . --view files -o files.svg
```

## CI and release contract

`agentpprof` is part of the AgentSight release surface. The CI pipeline should:

- build, clippy-check, and test `agentpprof` on pull requests;
- assign `agentpprof` the same release version as `agentsight`;
- update its `agent-session` dependency to the newly published
  `agent-session` version;
- build the release binary and upload it to the GitHub Release;
- publish the `agentpprof` crate to crates.io;
- smoke-test `agentpprof --version`, `agentpprof --help`, and
  `cargo install agentpprof --version <release>`.

This keeps the command usable as an independent tool, not only as an
AgentSight repository artifact.

## Troubleshooting

If no sessions are found, pass explicit `--session-file` paths and confirm the
session `cwd` matches `--project-root`.

If labels are too generic, add a few `--tag-rule` entries for the project. Do
not try to make every prompt unique. Good tags preserve useful semantic
diversity while merging meaningless long-tail fragments.

If pprof output opens but looks unfamiliar, remember that the sample unit is
not CPU time. Use `go tool pprof -top` to inspect the widest semantic frames,
then generate SVG or folded output when you need the full stack shape.

If a public artifact might contain sensitive information, prefer SVG, folded,
or pprof output, and do not pass `--include-previews`.
