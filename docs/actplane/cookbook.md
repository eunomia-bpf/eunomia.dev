# ActPlane Policy Examples

This cookbook turns the test policy corpus into user-facing examples. Each
example shows the problem, the rule, how to review it before enforcement, and
what violation to expect.

Start every policy with a static review:

```bash
actplane compile --policy test/policies/01_secret_no_exfil.yaml --explain
actplane doctor --policy test/policies/01_secret_no_exfil.yaml
```

Then enforce around a command:

```bash
sudo -E actplane --policy test/policies/01_secret_no_exfil.yaml run -- <agent-or-command>
```

Use `notify` policies or `actplane compile --explain` first when adapting a
policy to a real repository. Rules that use `block` need BPF-LSM support for
pre-operation denial; see [support-matrix.md](support-matrix.md).

## Example Map

| Scenario | Policy file | Main primitive |
| --- | --- | --- |
| Secret-derived data cannot leave the host | `test/policies/01_secret_no_exfil.yaml` | file source + network/file sinks |
| Untrusted input needs review before privileged action | `test/policies/02_prompt_injection_review.yaml` | endpoint/file source + endorsement |
| Production database access must pass through a tool | `test/policies/03_mediation_lineage.yaml` | lineage mediation |
| Agent writes stay inside the workspace | `test/policies/04_workspace_confinement.yaml` | lineage-scoped write block |
| Tests must run before commit | `test/policies/06_test_before_commit_since.yaml` | temporal gate + staleness |
| Review or audit subagent is read-only | `actplane init --template readonly-review` | subtree capability scope |
| Local secrets can be released only after redaction | `test/policies/08_secret_declassify.yaml` | declassification |
| Agent cannot bypass git policy through shell/Python | `test/policies/09_cross_tool_git.yaml` | process tree coverage |
| Release artifacts need human review | `test/policies/19_reviewed_release.yaml` | file source + endorsement |
| Child domains inherit parent policy | `test/policies/24_domain_session_auto_select.yaml` | domains |

## Secret-Derived Data Must Stay Local

Problem: an agent reads `.env` or `/etc/secrets/**`, then a later subprocess
tries to connect to the network or write into a shared export path.

```yaml
version: 1
policy: |
  source SECRET = file "**/.env"
  source SECRET = file "/etc/secrets/**"

  rule no-exfil:
    block connect endpoint "*" if SECRET
    block write file "/shared/**" if SECRET
    because "secret data must not leave the host"
```

Review:

```bash
actplane compile --policy test/policies/01_secret_no_exfil.yaml --explain
```

Expected violation: a process that has read a secret source and then connects
out receives a `no-exfil` match with the reason `secret data must not leave the
host`.

## Prompt-Injection Review Gate

Problem: task context comes from the network or downloads, and then the agent
tries to run privileged actions such as `git push` or deployment.

```yaml
version: 1
policy: |
  source UNTRUST = endpoint "*"
  source UNTRUST = file "**/downloads/**"

  rule no-injected-priv:
    block exec "git" "push" if UNTRUST and not REVIEWED
    block exec "**/deploy*" if UNTRUST and not REVIEWED
    because "untrusted input must not drive privileged actions"

  endorse REVIEWED by exec "**/human-approve"
```

Review:

```bash
actplane compile --policy test/policies/02_prompt_injection_review.yaml --explain
```

Expected violation: privileged actions derived from untrusted input are blocked
until an approved review command adds the `REVIEWED` label.

## Production Database Through Migration Tool

Problem: `prod.db` should be opened only by the migration tool, not by ad-hoc
agent scripts.

```yaml
version: 1
policy: |
  rule mediate-proddb:
    block open file "**/prod.db" unless lineage-includes exec "**/migrate"
    because "prod.db only via the migration tool"
```

Review:

```bash
actplane compile --policy test/policies/03_mediation_lineage.yaml --explain
```

Expected violation: direct opens of `prod.db` are blocked. Opens from a process
whose lineage includes `migrate` are allowed.

## Workspace Confinement

Problem: a coding agent can use `bash`, scripts, or generated programs to write
outside its assigned workspace.

```yaml
version: 1
policy: |
  source AGENT = exec "**/codex"

  rule confine-writes:
    block write file "/**" if AGENT unless target "/work/**"
    block unlink file "/**" if AGENT unless target "/work/**"
    because "agent may only modify /work"
```

Review:

```bash
actplane compile --policy test/policies/04_workspace_confinement.yaml --explain
```

Expected violation: writes or deletes outside `/work/**` from the agent process
tree are blocked.

## Fresh Tests Before Commit

Problem: the agent ran tests once, edited source again, and then committed with
stale test evidence.

```yaml
version: 1
policy: |
  source AGENT = exec "**/codex"

  rule test-before-commit:
    block exec "git" "commit" if AGENT unless after exec "**/pytest" since write "src/**" or write "tests/**"
    because "tests are stale after source edits"
```

Review:

```bash
actplane compile --policy test/policies/06_test_before_commit_since.yaml --explain
```

Expected violation: `git commit` from the agent tree is denied unless `pytest`
has run after the latest write to `src/**` or `tests/**`.

## Read-Only Review or Audit Subagent

Problem: a review or audit subagent should inspect evidence only. It should not
write files, delete files, or make repository changes.

```yaml
version: 1
policy: |
  source REVIEWER = exec "**/review-agent"

  rule readonly-review:
    block write file "/**" if REVIEWER
    block unlink file "/**" if REVIEWER
    block exec "git" if REVIEWER
    because "review domain is read-only"
```

Review:

```bash
actplane init --template readonly-review --out actplane.yaml
actplane compile --policy actplane.yaml --explain
```

Expected violation: any write, delete, or git execution in the review-agent
subtree reports `readonly-review`.

## Redaction Path for Secrets

Problem: secret-derived data should not leave the host unless it has passed
through a redactor.

```yaml
version: 1
policy: |
  source SECRET = file "**/.env"

  rule no-exfil:
    block connect endpoint "*" if SECRET
    because "redact secrets before egress"

  declassify SECRET by exec "**/redact"
```

Review:

```bash
actplane compile --policy test/policies/08_secret_declassify.yaml --explain
```

Expected violation: network egress after reading `.env` is blocked. A sanctioned
redaction command can clear the label for the redacted path.

## Cross-Tool Git Policy

Problem: a tool-layer rule can block a direct git tool call but miss
`bash -c git ...` or Python `subprocess.run(["git", ...])`.

```yaml
version: 1
policy: |
  source AGENT = exec "**/codex"

  rule no-git:
    block exec "git" if AGENT
    because "git is disabled for this agent"
```

Review:

```bash
actplane compile --policy test/policies/09_cross_tool_git.yaml --explain
```

Expected violation: any descendant process that executes `git` is blocked,
regardless of whether the path came through a tool call, shell, script, or SDK.

## Reviewed Release

Problem: a release artifact under `dist/**` should not be published by an agent
until a review step endorsed the session.

```yaml
version: 1
policy: |
  source AGENT = exec "**/codex"
  source RELEASE = file "dist/**"

  rule release-needs-review:
    block exec "gh" "release" if AGENT and RELEASE and not REVIEWED
    block connect endpoint "*" if AGENT and RELEASE and not REVIEWED
    because "release artifacts need human review before publishing"

  endorse REVIEWED by exec "**/review"
```

Review:

```bash
actplane compile --policy test/policies/19_reviewed_release.yaml --explain
```

Expected violation: release publication or external egress is blocked until a
review command adds `REVIEWED`.

## Parent and Child Domains

Problem: parent policy should apply to all children, while a helper domain adds
extra local rules for its own subtree.

```yaml
version: 1

rules:
  protect-policy:
    ifc: |
      source COMMAND = exec "**"
      rule protect-policy:
        kill write file "**/actplane.yaml" if COMMAND
        because "active policy is protected"

  read-only-helper:
    ifc: |
      source COMMAND = exec "**"
      rule read-only-helper:
        kill write file "/repo/**" if COMMAND
        because "helper domain is read-only"

domains:
  session:
    bind:
      - rule: protect-policy
        mode: locked

  helper:
    parent: session
    bind:
      - rule: read-only-helper
        mode: locked
```

Review:

```bash
actplane compile --policy test/policies/24_domain_session_auto_select.yaml --domains
actplane compile --policy test/policies/24_domain_session_auto_select.yaml --domain helper --explain
```

Expected violation: the helper domain inherits parent policy protection and adds
its local read-only rule. Child domains cannot remove locked parent rules.

## From Template to Policy

Many examples also exist as built-in templates:

```bash
actplane init --list-templates
actplane init --template test-before-commit --print
actplane init --template workspace-confinement --set writable_path=/repo/tasks/** --out actplane.yaml
actplane compile --explain --report-out docs/actplane-review.txt
```

Use templates for the first policy in a project, then promote tuned examples
into project-owned `actplane.yaml`.
