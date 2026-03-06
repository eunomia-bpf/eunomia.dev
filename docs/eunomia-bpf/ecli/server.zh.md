---
title: ecli server
catagories: ['ecli']
---

# ecli server

`ecli-server` 是 eunomia-bpf 的远程执行控制面，用来通过 HTTP 启动、停止、查看和管理远程主机上的 eBPF 程序。

## 功能

- 通过 HTTP 启动和停止 eBPF 程序
- 按游标拉取日志，或使用 `--follow` 持续跟踪
- 列出服务器上的运行任务
- 配合 `ecli client` 实现远程控制

## 安装

以 Ubuntu 为例：

```sh
# 下载独立的 server 二进制
wget https://github.com/eunomia-bpf/eunomia-bpf/releases/latest/download/ecli-server-ubuntu-latest.tar.gz
tar -xzf ecli-server-ubuntu-latest.tar.gz
chmod +x ./ecli-server

# 下载 ecli 客户端
wget https://aka.pw/bpf-ecli -O ecli
chmod +x ./ecli
```

## 启动服务

```console
$ sudo ./ecli-server
[2026-03-06T00:00:00Z] INFO Serving at 127.0.0.1:8527
```

默认监听地址是 `127.0.0.1:8527`。

## 远程执行流程

最常见的使用方式如下：

1. 在目标主机上启动 `ecli-server`
2. 在另一台机器或另一个 shell 中使用 `ecli client`
3. 启动任务、查看日志、停止任务

```console
$ ./ecli client --endpoint http://127.0.0.1:8527 start ./program.json
1

$ ./ecli client --endpoint http://127.0.0.1:8527 log 1
TIME     EVENT COMM             PID     PPID    FILENAME/EXIT CODE
16:03:16 EXEC  sh               51857   1711    /bin/sh
16:03:16 EXIT  sh               51857   1711    [0] (1ms)

$ ./ecli client --endpoint http://127.0.0.1:8527 log 1 --follow

$ ./ecli client --endpoint http://127.0.0.1:8527 list

$ ./ecli client --endpoint http://127.0.0.1:8527 stop 1
```

可以先看客户端子命令：

```console
$ ./ecli client --help
Client operations

Usage: ecli client [OPTIONS] <COMMAND>

Commands:
  start   Start an ebpf program on the specified endpoint
  stop    Stop running a task on the specified endpoint
  log     Fetch logs of the given task
  pause   Pause the task
  resume  Resume the task
  list    List tasks on the server
  help    Print this message or the help of the given subcommand(s)

Options:
  -e, --endpoint <ENDPOINT>  API endpoint [default: http://127.0.0.1:8527]
  -h, --help                 Print help
```

## 构建模式

如果你只需要远程控制，不需要本地运行 eBPF，可以构建一个更小的 `http` only 客户端：

```sh
cd ecli/client
cargo build --release --no-default-features --features http
```

这种模式适合 Windows 或其他只需要远程调用 `ecli-server` 的环境。

## 日志机制

服务端会为每个任务维护一份日志缓冲区：

- 每条日志都有时间戳游标
- `ecli client log <id>` 会拉取当前缓冲区日志
- `ecli client log <id> --follow` 会持续轮询新日志
- 客户端推进游标后，旧日志可以被清理

## HTTP API

OpenAPI 定义在 [`ecli/apis.yaml`](https://github.com/eunomia-bpf/eunomia-bpf/blob/master/ecli/apis.yaml)。

示例接口：

```http
POST /api/v1/ebpf/start
Content-Type: application/json

{
  "program_data": "<base64_or_json>",
  "args": ["--arg1", "value1"]
}
```

```http
GET /api/v1/ebpf/log/{id}?cursor={timestamp}
```

```http
GET /api/v1/ebpf/list
```

```http
POST /api/v1/ebpf/stop/{id}
```

也可以直接用 `curl` 调用：

```console
$ curl http://127.0.0.1:8527/task
{"tasks":[{"status":"running","id":3,"name":"bpf-program-1691432359"}]}

$ curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "id": 3,
    "log_cursor": 0,
    "maximum_count": 100
  }' \
  http://127.0.0.1:8527/log
```

## 安全说明

`ecli-server` 本身不提供内建认证或鉴权。
如果要暴露到 localhost 之外，请放在带认证能力的反向代理或网关之后。
