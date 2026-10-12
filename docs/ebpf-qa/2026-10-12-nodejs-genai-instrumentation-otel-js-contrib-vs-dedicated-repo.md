# Why does Node.js GenAI instrumentation live in opentelemetry-js-contrib rather than a dedicated repository, and why is there no Node.js equivalent to the Python GenAI instrumentation repo?

There is no standalone Node.js GenAI instrumentation repository; that is the deliberate state of the JS distribution, not a gap that will be closed by mirroring the Python layout. Node.js auto-instrumentation for GenAI client libraries ships from `opentelemetry-js-contrib` (the community-maintained instrumentation monorepo), and the GenAI semantic conventions are federated into a shared `semantic-conventions-genai` repository whose auto-generated code lands in the core `opentelemetry-js` client. The Python side instead got its own dedicated `opentelemetry-python-genai` repository. The asymmetry is not an oversight: the two language projects distribute contrib instrumentation from different homes (Python keeps per-signal instrumentation packages in a contrib repo; JS keeps them in one contrib monorepo), and the conventions layer is language-neutral, so a per-language GenAI instrumentation repo was only needed on the Python side, where the dedicated repo also carries the shared GenAI helper package.

## The mechanism

The distribution split is the boundary. OpenTelemetry's GenAI work is split into two independent layers that do not have to live in the same repository as the per-language instrumentation code:

1. **Conventions layer** — `semantic-conventions-genai` is the language-neutral source of truth for GenAI spans, metrics, events, and the attribute registry. It extends the core semantic-conventions repository and generates per-language reference material from its YAML model. This is shared across all languages, so it has no per-language home.
2. **Instrumentation layer** — this is where the asymmetry lives. Each language ships the instrumented SDK integrations from wherever that language keeps community instrumentation:
   - Python: the GenAI client-library instrumentations are consolidated in the dedicated `opentelemetry-python-genai` repo, which builds on the shared GenAI helper package and emits spans, metrics, and logs per the GenAI conventions. It covers the major GenAI clients (OpenAI, Anthropic, Bedrock, LangChain, Llama-Index, OpenAI Agents, and others) as separate packages with their own supported-version windows.
   - Node.js: GenAI instrumentation stays in `opentelemetry-js-contrib`, the single monorepo that already houses all community-maintained JS instrumentations. The OpenAI instrumentation there is one package among the many in that monorepo, wired through the standard `@opentelemetry/auto-instrumentations-node` bundle.

The conventions layer landing in the core `opentelemetry-js` client is what makes a dedicated JS GenAI repo unnecessary from the conventions side: the generated GenAI semantic-convention bindings ship with the core client, and the instrumentation packages that consume them are already organized into the contrib monorepo. There is no structural reason in the JS project to split GenAI instrumentations out of that monorepo the way Python did.

## Verification and debugging path

   - Node.js: GenAI instrumentation stays in `opentelemetry-js-contrib`, the single monorepo that already houses all community-maintained JS instrumentations. The OpenAI instrumentation there is one package among the many in that monorepo, wired through the standard auto-instrumentations bundle for Node.
2. Open `opentelemetry-js-contrib` and note that all community JS instrumentations - including the OpenAI GenAI one - are one package among the monorepo's many, distributed under the same contribution rules as every other instrumentation.
3. Open the `opentelemetry-python-genai` README: it lists a separate, per-client set of GenAI instrumentation packages that live in their own dedicated repo and are built on a shared GenAI helper package - the layout that has no Node.js counterpart.
4. Check the `opentelemetry-js` release notes: the GenAI semantic conventions federate into the shared conventions repo and their generated code ships in the core client, which is why the conventions side is not split per language.

## The limitation

- The asymmetry is a current-state description, not a guaranteed future layout: the JS project may later split GenAI instrumentations into their own repo, and the Python layout is itself still young, so "there is no Node.js equivalent repo" is a statement about where the code lives today, not a policy.
- "Lives in contrib" still means the same maturity boundary as every contrib instrumentation: it is community-maintained, not part of the core API/SDK distribution.
- The dedicated Python repo is not merely a different folder: it carries a shared GenAI helper package that the JS-side instrumentations do not have, so the two ecosystems are not interchangeable layouts and the Python repo is not simply the JS monorepo's future.
- Searching for a Node.js GenAI repo and finding none is not an access gap; the answer is the layout above, and the absence of a repo is the correct, expected result.

## References

- [opentelemetry-js-contrib](https://github.com/open-telemetry/opentelemetry-js-contrib) - the single monorepo for community-maintained JS instrumentation, where the GenAI/OpenAI instrumentation lives.
- [opentelemetry-instrumentation-openai for Node.js](https://github.com/open-telemetry/opentelemetry-js-contrib/blob/main/packages/instrumentation-openai/README.md) - the OpenAI auto-instrumentation package inside that monorepo and its auto-instrumentations bundle.
- [opentelemetry-python-genai](https://github.com/open-telemetry/opentelemetry-python-genai) - the dedicated Python GenAI instrumentation repo, with per-client packages built on a shared GenAI helper.
- [semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai) - the language-neutral GenAI conventions repo that the generated per-language bindings in opentelemetry-js come from.
- [semantic-conventions-genai issue 247](https://github.com/open-telemetry/semantic-conventions-genai/issues/247) - the discussion of which artifacts should be generated for a language and whether they belong in separate repos.

## Community discussion today

Two opt-in archive channels (the two CNCF OpenTelemetry instrumentation channels) produced fourteen messages this run. The thread this page answers: a practitioner asked whether there were plans for a Node.js repository similar to the Python GenAI instrumentation repo; the reply was that, as of now, there were not known plans for one, and that the GenAI instrumentation was landing in `opentelemetry-js-contrib` with the auto-generated conventions landing in `opentelemetry-js`, with issue 247 being the only guidance visible on what artifacts to generate and whether they should live in separate repos. The other threads in the window had already been answered by earlier posts - the available-skills span-attribute modeling question, the `logger.enabled` opt-in/opt-out question, and the model-serving-signals catalog question - so this run selected the Node.js repository-organization boundary, which had not been answered on this site.

Channel coverage for this run: the two opt-in archives supplied fourteen messages. The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.
