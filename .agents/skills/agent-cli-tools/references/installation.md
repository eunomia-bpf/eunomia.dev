# Installation And Local State Reference

Use this file only when installing, updating, repairing PATH, or debugging authentication for local agent CLIs.

## Current Machine Snapshot

Captured on 2026-07-18 America/Los_Angeles.

Installed and verified:

- Claude Code: `C:\Users\10678\.local\bin\claude.exe`, version `2.1.214`.
- Kimi Code: `C:\Users\10678\.kimi-code\bin\kimi.exe`, version `0.27.0`.
- Grok Build: `C:\Users\10678\.grok\bin\grok.exe`, version `0.2.103`.
- Grok alias: `C:\Users\10678\.grok\bin\agent.exe`, same version as `grok`.

Windows user PATH entries added:

```text
C:\Users\10678\.local\bin
C:\Users\10678\.kimi-code\bin
C:\Users\10678\.grok\bin
```

Authentication and minimal calls were verified:

```powershell
claude auth status
claude -p "Reply with exactly: ok"
kimi doctor
kimi -p "Reply with exactly: ok"
grok -p "Reply with exactly: ok" --no-alt-screen
```

`codex` was not found in PATH at the time this reference was written. Re-check with:

```powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path','User') + ';' + [Environment]::GetEnvironmentVariable('Path','Machine')
where.exe codex
```

## Official Install Commands

Always prefer official installer pages or package names. Avoid similarly named third-party packages.

### Claude Code

Official Windows PowerShell installer:

```powershell
irm https://claude.ai/install.ps1 | iex
```

Useful checks:

```powershell
claude --version
claude auth login
claude auth status
```

### Kimi Code

Official Windows PowerShell installer:

```powershell
irm https://code.kimi.com/kimi-code/install.ps1 | iex
```

Kimi Code on Windows expects Git for Windows / Git Bash. If Git is installed outside the default location, set `KIMI_SHELL_PATH` to the absolute `bash.exe` path.

Useful checks:

```powershell
kimi --version
kimi login
kimi doctor
```

### Grok Build

Official installer is a Bash script. On Windows, run it from Git for Windows / MSYS2 Bash:

```bash
curl -fsSL https://x.ai/cli/install.sh | bash
```

Then add this to Windows user PATH if the installer only updated Git Bash startup files:

```text
%USERPROFILE%\.grok\bin
```

Useful checks:

```powershell
grok --version
grok update --check --json
grok login --device-code
grok -p "Reply with exactly: ok" --no-alt-screen
```

Observed on 2026-07-18 in `C:\Users\10678\OneDrive\文档\eunomia.dev`: Grok Build `0.2.103` was the latest stable version according to `grok update --check --json`, but a real review attempt emitted `tool_output_error` for `read_file` and prompt-file retries returned planning text without findings. Treat that as a tool failure, not a valid review, and switch to Claude or Kimi after one prompt-file retry.

### OpenAI Codex CLI

The official Codex CLI page says to install Codex, open a project directory, run `codex`, sign in on first run, then start with a prompt such as `Tell me about this project`.

Official macOS/Linux standalone installer shown in the docs:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

For Windows or npm-based installation, open the official Codex CLI docs and use the current Windows or npm tab. Recent official changelog entries publish the scoped npm package form:

```powershell
npm install -g @openai/codex@latest
```

Do not install the unscoped `codex` package.

Useful first-run commands after install:

```powershell
codex --version
codex
codex exec "Tell me about this project"
```

Inside the interactive CLI, useful slash commands include `/init`, `/status`, `/permissions`, `/model`, and `/review`.

## Source Links

- Claude Code setup: https://code.claude.com/docs/en/setup
- Kimi Code CLI getting started: https://moonshotai.github.io/kimi-cli/en/guides/getting-started.html
- Kimi API Platform with Kimi Code CLI: https://platform.kimi.ai/docs/guide/kimi-code-cli
- Grok Build CLI: https://x.ai/cli
- xAI API quickstart: https://docs.x.ai/developers/quickstart
- Codex CLI: https://learn.chatgpt.com/docs/codex/cli
- Codex changelog: https://learn.chatgpt.com/docs/changelog
