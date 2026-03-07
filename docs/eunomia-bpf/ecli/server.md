---
title: ecli server
catagories: ['ecli']
---

# Legacy ecli server mode

`ecli-server` and the `ecli client` subcommand were removed from the main branch in March 2026 to reduce maintenance overhead.

Current releases no longer ship:

- `ecli-server` binaries
- the `ecli client` subcommand
- the `http`-only client build mode

## What to use now

For maintained workflows, use:

- `ecli run` to execute a local package, URL, OCI image, or Wasm module on the same machine
- `ecli pull` to fetch OCI images locally before inspection or execution
- `ecli push` to publish Wasm modules to an OCI registry

The historical GitHub Pages URLs under `https://eunomia-bpf.github.io/eunomia-bpf/...` are still supported for local `ecli run` compatibility. What was removed here is only the old remote HTTP control plane.

Example:

```bash
wget https://aka.pw/bpf-ecli -O ecli
chmod +x ./ecli
sudo ./ecli https://eunomia-bpf.github.io/eunomia-bpf/sigsnoop/package.json
sudo ./ecli run ghcr.io/eunomia-bpf/execve:latest
```

## Archived implementation

The last implementation of the remote HTTP mode is preserved on the `archive/ecli-remote-http` branch of the main repository:

- https://github.com/eunomia-bpf/eunomia-bpf/tree/archive/ecli-remote-http/ecli

That archived branch is the right place to look if you need the historical `ecli-server`, the old OpenAPI surface, or the `http`-only client build.

## Notes for existing users

- Existing documentation or blog posts that mention `ecli-server` describe historical behavior.
- If you need remote orchestration today, keep `ecli` on the target host and use your own SSH, container, or job-control layer around `ecli run`.
