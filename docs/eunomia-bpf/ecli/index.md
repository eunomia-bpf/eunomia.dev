---
title: ecli
catagories: ['ecli']
---

# ecli: run, pull, and publish eunomia-bpf programs

`ecli` is the local eunomia-bpf CLI for loading, running, pulling, and pushing precompiled programs.

The legacy remote HTTP mode (`ecli client` / `ecli-server`) has been removed from the main branch to reduce maintenance overhead. The last implementation is preserved on the `archive/ecli-remote-http` branch.

## Usage

```sh
sudo ecli <COMMAND>
```

## Examples

Run the eBPF program as wasm or json.

```sh
# run with wasm bpf modules
sudo ecli run runqlat.wasm
# run with json bpf object only
sudo ecli run package.json
```

Or run the eBPF program as a tar file that contains minimal BTF info and a BPF object.

```sh
sudo ecli run client.tar
```

The `ecc` packaged tar contains custom BTF files and `package.json`, which can be run on older kernels.

For details, see [ecc-btfgen](../ecc/usage.md#options)

## Commands

- run - Run the ebpf program as tar or json.
- push - Push a container to an OCI registry.
- pull - Pull a container from an OCI registry.

## Install

```bash
wget https://aka.pw/bpf-ecli -O ecli
chmod +x ./ecli
```

## GitHub Pages and URL examples

The historical GitHub Pages workflow is still kept for compatibility:

```bash
sudo ./ecli https://eunomia-bpf.github.io/eunomia-bpf/sigsnoop/package.json
sudo ./ecli run https://eunomia-bpf.github.io/eunomia-bpf/sigsnoop/app.wasm
```

## OCI examples

Pull an image locally:

```bash
./ecli pull ghcr.io/eunomia-bpf/execve:latest
```

Run directly from the registry:

```bash
sudo ./ecli run ghcr.io/eunomia-bpf/execve:latest
```

Push a Wasm module:

```bash
./ecli push --module app.wasm ghcr.io/yourorg/mytool:v1.0
```

For historical notes about the removed remote mode, see [ecli server](server.md).
