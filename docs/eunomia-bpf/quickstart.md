---
title: quickstart
catagories: ['misc']
---

# Quick Start

- Github Template: [eunomia-bpf/eunomia-template](https://github.com/eunomia-bpf/eunomia-template)
- example bpf programs: [examples/bpftools](https://github.com/eunomia-bpf/eunomia-bpf/blob/master/examples/bpftools)
- tutorial: [eunomia-bpf/bpf-developer-tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial)

You can get pre-compiled eBPF programs running from an OCI registry to the kernel in `1` line of bash:

```bash
# download the latest release from GitHub Releases
$ wget https://github.com/eunomia-bpf/eunomia-bpf/releases/latest/download/ecli -O ecli && chmod +x ./ecli
$ sudo ./ecli run ghcr.io/eunomia-bpf/execve:latest
```

The legacy remote HTTP mode (`ecli client` / `ecli-server`) has been removed from the main branch and archived on `archive/ecli-remote-http`.
