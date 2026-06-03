# 使用说明

[English](https://github.com/eunomia-bpf/agentsight/blob/master/docs/usage.md) | **中文**

## 从源代码编译

### 1. 克隆仓库并初始化子模块

```sh
git clone https://github.com/eunomia-bpf/agentsight.git
cd agentsight
git submodule update --init --recursive
```

如果你已经克隆过仓库但尚未初始化子模块（`libbpf/` 和 `bpftool/` 目录为空），请执行：

```sh
git submodule update --init --recursive
```

### 2. 安装系统依赖

```sh
make install
```

这会安装编译所需的 libelf、zlib、clang、llvm、Node.js 和 Rust 工具链。

### 3. 编译

```sh
make build
```

编译成功后，agentsight 二进制程序生成在 `collector/target/release/agentsight`。

也可以单独编译各组件：

```sh
make build-bpf    # 仅编译 eBPF C 程序
make build-rust   # 仅编译 Rust collector
make build-frontend  # 仅编译前端
```

## 从源码运行

`make build` 完成后，在仓库根目录运行下面的命令。需要加载 eBPF probes
的命令推荐显式使用 `sudo`；AgentSight 在你忘记 sudo 时可以自动请求提权，
但那只是补救路径。

```sh
# 实时查看本机智能体 session
sudo ./collector/target/release/agentsight top

# 启动并记录一个命令
sudo ./collector/target/release/agentsight record -- claude

# 附加到已经运行的进程族
sudo ./collector/target/release/agentsight record -c claude

# 可配置的底层调试追踪
sudo ./collector/target/release/agentsight debug trace --server -c claude

# 原始 SSL 调试捕获，启用 HTTP 解析
sudo ./collector/target/release/agentsight debug ssl --http-parser
```

## record 与 debug trace 子命令对比

agentsight 提供 `record` 和 `debug trace` 两个主要追踪入口，它们共用底层执行逻辑，但面向不同的使用场景。

### record — 开箱即用的智能体录制

适用于快速录制 AI 智能体（Claude Code、Python AI 工具等）的行为，无需关心细节配置。

- `record -- <command>` 用于启动并记录一个命令；`record -c/-p` 用于附加到已运行进程
- **自动开启**：SSL 监控 + 进程监控 + 系统监控 + Web 服务器（端口 7395）
- **内置过滤规则**：自动过滤掉注册请求（`/v1/rgstr`）、HEAD 请求、空响应体、202 状态码、二进制数据等噪音
- 默认**静默模式**（不输出到控制台），数据写入 `record.log`
- 默认开启**日志轮转**

典型用法：

```sh
sudo ./agentsight record -- claude
```

### debug trace — 完全可控的灵活监控

适用于需要自定义监控范围、过滤规则的调试和分析场景。

- **无必填参数**，所有功能独立开关
- SSL（`--ssl`）、进程（`--process`）默认开启，但可关闭
- 系统监控（`--system`）、stdio 捕获（`--stdio`）、Web 服务器（`--server`）默认**关闭**，需手动开启
- 过滤规则完全由用户通过 `--ssl-filter`、`--http-filter` 自定义
- 默认输出到控制台，可用 `-q` 静默

典型用法：

```sh
sudo ./agentsight debug trace --ssl true --process false --server true --http-filter "request.method=POST"
```

### 对比总结

| 维度 | record | debug trace |
|------|--------|-------|
| 定位 | 一键录制，预设优化 | 灵活定制，精细控制 |
| 必填参数 | 无；可用 `-- <command>`、`-c <comm>` 或 `-p <pid>` | 无 |
| Web 服务器 | 始终开启 | 需 `--server true` |
| 系统监控 | 始终开启 | 需 `--system true` |
| 控制台输出 | 默认关闭 | 默认开启 |
| 过滤规则 | 内置预设 | 用户自定义 |
| 日志轮转 | 默认开启 | 需 `--rotate-logs` |

简单来说：**日常录制用 `record`，深度调试用 `debug trace`**。
