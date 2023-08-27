# bpf-compatible

## Brief description

This repo contains a set of toolchain to simplify the building and running eBPF program on kernels without native BTF support.
It utilitizes [btfhub](https://github.com/aquasecurity/btfhub-archive) to drop the depencency of native BTF

## What's in it

This repo mainly contains three parts:

- A shell script `script/btfgen`, which can be used to clone the `btfhub` repo, or create stripped btf based on the compiled eBPF program and pack the btf archives into `.tar.gz`
- A Rust crate `bpf-compatible-rs`, which were used by [eunomia-bpf](https://github.com/eunomia-bpf/eunomia-bpf/) to implement the unpacking and loading of the package that `btfgen` generated
- A Rust crate `bpf-compatible-sys` and its C binding `btf_helpers.h`, which could be linked to other programs. It implements the unpacking and loading of the tar archive that `btfgen` generates. It can load tar archive either embedded into the executable, or provided by an external source. Used together with the `btf_helpers.h`, it can conveniently modify `struct bpf_object_open_opts*` and set the `custom_btf_path`.

## Usage - Manually use the toolchain

Usually the `prepare` steps could only be run once.

### Prepare - `btfhub-archive`

You will need a git repo like [btfhub-archive](https://github.com/aquasecurity/btfhub-archive), which contains prebuilt btf archive of various releases, archs, and kernels. We also provided a repo for demonstrating only (It contains a little number of kernel btf archives) [https://github.com/eunomia-bpf/btfhub-archive](https://github.com/eunomia-bpf/btfhub-archive).

The repo should have the structure like:
```plain
|- ubuntu <ID in os-release>
|- ---- 22.04 <VERSION in os-release>
|- ---- ---- x86_64 <machine in uname>
|- ---- ---- ---- 5.15.0-71-generic.btf <kernel-release in uname>
```
- Note: words in `<>` are explanation of the folder name.

### Prepare - build `bpf-compatible-sys`

Run `make` in the `bpf-compatible-sys` folder. It will build `libbpf_compatible.a` for you, which is a static library used to linked to libbpf programs

### Prepare - Use `btfgen` to fetch `btfhub-archive`

Run `./script/btfgen fetch` to download the `https://github.com/aquasecurity/btfhub-archive` repo to `~/.cache/eunomia/btfhub`. You can use `BTFHUB_REPO_URL` to override the repo url, or use `BTFHUB_CACHE_DIR` to override the local directory.

### Write your kernel program

Since generating the btf tar requires the compiled kernel program, so you should provide that first. 

### Create a btf tar archive with `btfgen`

Run `./script/btfgen btfgen xxx.o -o min_core_btfs.tar.gz` to pack the tailored btf archive into `min_core_btfs.tar.gz`. `xxx.o` is the name of the compiled kernel program.

### Create a linkable object of the btf archive

Run `ld -r -b binary min_core_btfs.tar.gz -o min_core_btfs_tar.o` to generate a linkable `min_core_btfs_tar.o`. This file declares symbols named `_binary_min_core_btfs_tar_gz_start` and `_binary_min_core_btfs_tar_gz_end`, indicating the range of the embed tar.gz file

### Write the userspace program with `btf_helpers.h`

Call `int ensure_core_btf(struct bpf_object_open_opts*)` before opening the skeleton. For example:
```c
	libbpf_set_print(libbpf_print_fn);

	err = ensure_core_btf(&open_opts);
	if (err) {
		fprintf(stderr, "failed to fetch necessary BTF for CO-RE: %s\n", strerror(-err));
		return 1;
	}

	obj = execsnoop_bpf__open_opts(&open_opts);
	if (!obj) {
		fprintf(stderr, "failed to open BPF object\n");
		return 1;
	}
```

And call `void clean_core_btf(struct bpf_object_open_opts*)` before exiting. For example:
```c
cleanup:
	cleanup_core_btf(&open_opts);
```

### Link your userspace program, `libbpf_compatible.a`, and `min_core_btfs_tar.o` together

It can be directly done by calling `clang <your_program> libbpf_compatible.a min_core_btf.tar.o`

## Usage - more simply

We have adapted the `libbpf-bootstrap` to the `bpf-compatible` toolchain. So there is a more simpler way:
- Put your `xxx.c` (userspace space program) and `xxx.bpf.c`(kernel program) in the `example/c` folder, or directly modify an exist one
- Add the name (`xxx` in the last row) to line 27 of `example/c/Makefile`, e.g `APPS = bootstrap execsnoop xxx`
- Run `make xxx` in `example/cs`
