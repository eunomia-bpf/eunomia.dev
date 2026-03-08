---
title: ecli server
catagories: ['ecli']
---

# 旧版 ecli server 模式

`ecli-server` 和 `ecli client` 子命令已在 2026 年 3 月从主分支移除，以降低维护复杂度。

当前发布版本已经不再提供：

- `ecli-server` 二进制
- `ecli client` 子命令
- `http` only 客户端构建模式

## 现在应该怎么用

当前维护中的工作流是：

- 用 `ecli run` 在本机运行本地包、URL、OCI 镜像或 Wasm 模块
- 用 `ecli pull` 先把 OCI 镜像拉到本地再检查或执行
- 用 `ecli push` 将 Wasm 模块发布到 OCI 仓库

`https://eunomia-bpf.github.io/eunomia-bpf/...` 下面的历史 GitHub Pages URL 仍然保留给本地 `ecli run` 兼容使用；这里移除的只是旧的远程 HTTP 控制面。

示例：

```bash
wget https://aka.pw/bpf-ecli -O ecli
chmod +x ./ecli
sudo ./ecli https://eunomia-bpf.github.io/eunomia-bpf/sigsnoop/package.json
sudo ./ecli run ghcr.io/eunomia-bpf/execve:latest
```

## 归档实现

远程 HTTP 模式最后一版实现保存在主仓库的 `archive/ecli-remote-http` 分支中：

- https://github.com/eunomia-bpf/eunomia-bpf/tree/archive/ecli-remote-http/ecli

如果你需要查找历史上的 `ecli-server`、旧版 OpenAPI 接口或 `http` only 客户端，请直接看这条归档分支。

## 给现有用户的说明

- 仍然提到 `ecli-server` 的文档或博客，描述的都是历史行为。
- 如果你今天仍然需要远程编排，建议把 `ecli` 放在目标主机上，再在外层使用你自己的 SSH、容器或作业调度方案包装 `ecli run`。
