---
title: quickstart
catagories: ['misc']
---

# Quick Start

- Github Template: [eunomia-bpf/eunomia-template](https://github.com/eunomia-bpf/eunomia-template)
- example bpf programs: [examples/bpftools](https://github.com/eunomia-bpf/eunomia-bpf/blob/master/examples/bpftools)
- tutorial: [eunomia-bpf/bpf-developer-tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial)

You can get pre-compiled eBPF programs running from GitHub Pages URLs or an OCI registry in `1` line of bash:

```bash
# download the latest release (aka.pw/bpf-ecli redirects to the current GitHub release asset)
$ wget https://aka.pw/bpf-ecli -O ecli && chmod +x ./ecli
$ sudo ./ecli https://eunomia-bpf.github.io/eunomia-bpf/sigsnoop/package.json # historical GitHub Pages workflow, kept for compatibility
$ sudo ./ecli run ghcr.io/eunomia-bpf/execve:latest
```

The legacy remote HTTP mode (`ecli client` / `ecli-server`) has been removed from the main branch and archived on `archive/ecli-remote-http`.
