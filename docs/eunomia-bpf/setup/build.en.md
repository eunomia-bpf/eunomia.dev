---
title: build from source
catagories: ['installation']
---

# build eunomia-bpf project

If you want to run the cli, in most cases, you don't need to build your own.

## Install Dependencies

You will need `clang`, `libelf` and `zlib` to build the examples, package names may vary across distros.

On Ubuntu/Debian, you need:

```shell
apt install clang libelf1 libelf-dev zlib1g-dev
```

On CentOS/Fedora, you need:

```shell
dnf install clang elfutils-libelf elfutils-libelf-devel zlib-devel
```

install rust toolchain

```shell
curl https://sh.rustup.rs -sSf | sh -s
```

## build bpf-loader-rs

bpf-loader is our core library, written in Rust. It has no other dependencies except `libelf` and `zlib`.

```shell
make bpf-loader-rs # build bpf-loader-rs
```

Note that this target runs the crate's `install-deps` step, which calls `apt-get` through `sudo`.

## build ecli

The cli tool is written in Rust as well. It links against `bpf-loader-rs` through a path dependency, so this builds both. It additionally needs OpenSSL and `pkg-config` (its own `install-deps` installs `libssl-dev` and `pkg-config` through `apt`):

```shell
make ecli
```

The resulting binary is `ecli/target/release/ecli-rs`. `make release` at the project root packages it under the name `ecli`, together with `ecc`, into `eunomia.tar.gz`:

```shell
make release
```

## build compiler

```shell
make ecc
```

`ecc` fetches and builds `bpftool` and a `vmlinux.h` set of its own at build time, so it needs network access and `libclang` (for `bindgen`):

```shell
apt install libclang-dev
```

In a network-isolated environment, set `SANDBOX` to a non-empty value to skip the fetching; the build then uses the `BPFTOOL_DIR` and `VMLINUX_DIR` directories you point it at.

The `bpftool` binary that `ecc` builds its workspace from comes from the `third_party/bpftool` submodule, and the `vmlinux.h` headers the compiler includes come from `third_party/vmlinux`, so a fresh clone needs them populated first:

```shell
git submodule update --init --recursive
```

The resulting binary is `compiler/cmd/target/release/ecc-rs`; `make release` packages a copy of it as `ecc-rs`.

## build wasm programs

`ecc` can compile C to wasm. That needs WASI SDK: download a [wasi-sdk](https://github.com/CraneStation/wasi-sdk/releases) release and extract it to `/opt/wasi-sdk`. The `wasm-runtime` submodule was removed from the project, so there is no separate `make` target for it any more.

# install from package manager

## nix

On any distros with nix installed:

### build ecc

```shell
nix build github:eunomia-bpf/eunomia-bpf#ecc # or ecli, see `nix flake show` for details # or ecli
```

Run application directly with:

```shell
nix run github:eunomia-bpf/eunomia-bpf#ecc -- -h
```

## openEuler

```shell
sudo dnf search eunomia-bpf
# and install
```


## more details

- The top-level [Makefile](https://github.com/eunomia-bpf/eunomia-bpf/blob/master/Makefile) shows what each target delegates to; the per-component ones ([`compiler/Makefile`](https://github.com/eunomia-bpf/eunomia-bpf/blob/master/compiler/Makefile), [`ecli/Makefile`](https://github.com/eunomia-bpf/eunomia-bpf/blob/master/ecli/Makefile)) have the details.
- You may want to refer to our CI for more build info: [ecc-binary.yml](https://github.com/eunomia-bpf/eunomia-bpf/blob/master/.github/workflows/ecc-binary.yml) and [ecli-binary.yaml](https://github.com/eunomia-bpf/eunomia-bpf/blob/master/.github/workflows/ecli-binary.yaml).
