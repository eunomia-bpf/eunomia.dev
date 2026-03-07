---
title: github template
catagories: ['ecc']
---

# Github Action 模板

`eunomia-template`：使用 GitHub Actions 编译 eBPF 程序，并将生成的 `package.json` 发布为 Release 资产。

请参考：https://github.com/eunomia-bpf/eunomia-template

# A template for eunomia-bpf programs

This is a template for eunomia-bpf eBPF programs. You can use it as a template, compile it online with `GitHub Actions`, or build it offline.

### Compile and run the eBPF code as simple as possible!

Download the pre-compiled `ecli` binary from here: [eunomia-bpf/eunomia-bpf](https://github.com/eunomia-bpf/eunomia-bpf/releases)

To install, just download and use the `ecli` binary from here: [eunomia-bpf/eunomia-bpf](https://github.com/eunomia-bpf/eunomia-bpf/releases):

```console
wget https://github.com/eunomia-bpf/eunomia-bpf/releases/latest/download/ecli -O ecli && chmod +x ecli
```

## use this repo as a GitHub template for online compilation

1. use this repo as a github template: see [creating-a-repository-from-a-template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)
2. modify `src/template.bpf.c`, commit it, and wait for the `publish.yml` workflow to finish
3. download the generated `src/package.json` asset from the latest release of your fork and run it locally:

```console
$ sudo ./ecli run package.json
```

## quick start

just write some code in the `bootstrap.bpf.c`, after that, simply run this:

```shell
$ docker run -it -v /path/to/repo:/src ghcr.io/eunomia-bpf/ecc-`uname -m`:latest # use absolute path
```

you will get a `package.json` in your root dir. Just run:

```shell
$ sudo ./ecli run package.json
```

The ebpf compiled code can run on different kernel versions(CO-RE). You can just copied the json to another machine.
see: [github.com/eunomia-bpf/eunomia-bpf](https://github.com/eunomia-bpf/eunomia-bpf) for the runtime, and [eunomia-bpf/eunomia-cc](https://github.com/eunomia-bpf/eunomia-cc) for our compiler tool chains.

## The code here

This is an example of eBPF code. The template currently ships `src/template.bpf.c` and `src/template.h`; adjust those files for your own program and let the workflow publish the resulting `package.json`.

## more examples

for more examples, please see: [eunomia-bpf/eunomia-bpf/tree/master/examples/bpftools](https://github.com/eunomia-bpf/eunomia-bpf/tree/master/examples/bpftools)
