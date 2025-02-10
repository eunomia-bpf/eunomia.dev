---
date: 2023-07-11
---

# 快速构建 eBPF 项目和开发环境，一键在线编译运行 eBPF 程序

如果您正在探索 eBPF 技术，并且对于如何开始搭建环境以及选择编程语言感到困惑，那么您来对地方了！本文将介绍一系列 GitHub 模板和 GitHub Codespaces，让您可以快速启动全新的 eBPF 项目，一键在线编译运行 eBPF 程序。现在就跟随我们的步骤，加速您的 eBPF 开发吧！
<!-- more -->

## **eBPF：探索 Linux 内核的新世界**

eBPF (Extended Berkeley Packet Filter) 是一种 Linux 内核技术，它可以在运行时动态加载并执行一些小的程序，以增强内核的网络、安全、性能和观测等方面的功能。eBPF 能够处理各种数据包，如 TCP/IP、UDP、ICMP 等，同时支持用户态和内核态编程，是一个相当强大的工具。

不过，要从零开始开发一个 eBPF 项目并不容易，需要熟悉内核的相关知识和各种工具链，同时还需要考虑性能、安全等方面的问题。这也是很多开发者在接触 eBPF 时所面临的挑战。

## **如何快速搭建环境以及选择编程语言？**

面对创建一个 eBPF 项目，您是否对如何开始搭建环境以及选择编程语言感到困惑？别担心，我们为您准备了一系列 GitHub 模板，以便您快速启动一个全新的 eBPF 项目。只需在 GitHub 上点击 `Use this template` 按钮，即可开始使用。

## **GitHub 模板：快速启动 eBPF 项目**

当您开始创建一个 eBPF 项目时，环境搭建和基础设置是必要且繁琐的。为了帮助您节省时间和精力，我们为您准备了一系列 GitHub 模板覆盖了不同的编程语言和框架，以满足您的不同需求。这些模板可让您快速启动一个全新的 eBPF 项目，而无需从头开始构建项目。

- [https://github.com/eunomia-bpf/libbpf-starter-template](http://link.zhihu.com/?target=https%3A//github.com/eunomia-bpf/libbpf-starter-template)：基于 C 语言和 libbpf 框架的 eBPF 项目模板
- [https://github.com/eunomia-bpf/cilium-ebpf-starter-template](http://link.zhihu.com/?target=https%3A//github.com/eunomia-bpf/cilium-ebpf-starter-template)：基于 C 语言和 cilium/ebpf 框架的 eBPF 项目模板
- [https://github.com/eunomia-bpf/libbpf-rs-starter-template](http://link.zhihu.com/?target=https%3A//github.com/eunomia-bpf/libbpf-rs-starter-template)：基于 Rust 语言和 libbpf-rs 框架的 eBPF 项目模板
- [https://github.com/eunomia-bpf/eunomia-template](http://link.zhihu.com/?target=https%3A//github.com/eunomia-bpf/eunomia-template)：基于 C 语言和 eunomia-bpf 框架的 eBPF 项目模板

这些启动模板包含以下功能：

- 一个 Makefile，让您可以一键构建项目
- 一个 Dockerfile，用于为您的 eBPF 项目自动创建一个容器化环境并发布到 Github Packages
- GitHub Actions，用于自动化构建、测试和发布流程
- eBPF 开发所需的所有依赖项

通过将现有仓库设置为模板，您和其他人可以快速生成具有相同基础结构的新仓库，从而省去了手动创建和配置的繁琐过程。GitHub 模板提供了一种简单的方式，让开发者可以快速启动一个新的项目，并且能够快速地获得与其他开发者共享的最佳实践和经验。

一旦您的仓库被设置为模板，其他用户就可以通过以下步骤创建一个新的仓库：

1. 打开您的模板仓库页面。
2. 点击 "Use this template" 按钮。
3. 输入新仓库的名称和描述，并选择要创建新仓库的组织或个人账户。
4. 单击 "Create repository from template" 按钮即可创建新的仓库。

!<https://picx.zhimg.com/80/v2-9147b573ee3df2d0f955fc62fb81128b_1440w.webp?source=d16d100b>

（如果自动发布镜像到 Github Packages 的 CI 失败，可能需要需要在仓库的 Settings 中配置 actions 权限，请参考后文）

通过 GitHub 模板，您可以快速启动一个新的项目，并且能够快速地获得与其他开发者共享的最佳实践和经验，从而更加专注于您的项目的核心功能和逻辑，提高开发效率和代码质量。在 Github template 中，您可以选择使用 C 语言、Rust 语言或者 eunomia-bpf 框架来开发 eBPF 程序，同时还可以使用 Github Actions 自动化构建、测试和发布 eBPF 程序二进制和容器镜像。更详细的信息，请参考官方文档：[https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-template-repository](http://link.zhihu.com/?target=https%3A//docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-template-repository)

## **GitHub Codespaces：云端在线编译 eBPF 程序**

GitHub Codespaces 是一个基于云的开发环境，能够为您提供快速、灵活的编译和运行 eBPF 程序的能力。不再需要在本地安装和配置各种软件，只需要使用 GitHub 的基础服务，就可以轻松地构建和调试 eBPF 程序。

在 Codespaces 中，每个代码仓库都有一个对应的 Docker 容器，容器中包含了您项目所需的所有依赖项和工具。您可以通过浏览器、Visual Studio Code 或者 JetBrains Gateway 等工具连接到 Codespaces，从而获得一个完全可配置的云端开发环境。

使用 GitHub Codespaces 进行 eBPF 程序的开发非常简单，只需要按照以下步骤进行即可：

1. 打开您的 eBPF 项目仓库。
2. 点击 "Code" 按钮，然后选择 "Open with Codespaces" 选项。

![Open with Codespaces](https://picx.zhimg.com/80/v2-c97afd28cc88ad73836eb4561267021c_1440w.webp)

1. GitHub 将会为您创建一个新的 Codespace，其中包含了您项目所需的所有依赖项和工具，这可能需要几分钟的时间，具体取决于您的网络速度和仓库的大小.
2. 一旦您的 Codespace 准备好使用，您就可以通过打开终端并导航到您的项目目录中，编译和运行您的 eBPF 程序。

![codespace](https://pic1.zhimg.com/80/v2-8b37f9241c284ac494555149272d1e57_1440w.webp?source=d16d100b>)

通过 GitHub Codespaces，您可以摆脱环境配置和版本兼容性等问题，使用最新的开发工具和框架，无需担心系统硬件限制和性能问题，而且可以在任何地方、任何设备上进行开发，无需受限于本地设备和网络速度。使用 Codespaces，您可以专注于代码编写和项目的核心业务，提高开发效率和代码质量，从而更加快速、高效地推动项目的进展。更详细的信息，请参考：[https://github.com/codespaces](http://link.zhihu.com/?target=https%3A//github.com/codespaces)

## **使用 docker 一键运行 eBPF 程序**

在 codespace 编写代码，提交后，Github Actions 会进行编译并自动发布容器镜像。注意，要使用自动发布镜像的功能，需要在仓库的 Settings 中配置 actions 权限：

![actions](https://picxzhimg.com/80/v2-2e0f9fc6aa0d1aee4231963432105626_1440w.webp?source=d16d100b>)

接下来，你可以在任何地方使用 docker 一键运行这个 eBPF 程序，例如：

`sudo docker run --rm -it --privileged ghcr.io/eunomia-bpf/libbpf-rs-template:latest`

![dockerdocker](https://pic1.zhimg.com/80/v2-ede596564dc3a701889ed161dcda9eb5_1440w.webp?source=d16d100b)

## **结语**

在 eBPF 项目的开发中，我们介绍了如何快速搭建环境以及选择编程语言，以及如何使用 GitHub 模板来启动项目，大大节省了开发者的时间和精力，让开发者专注于创造核心功能和业务逻辑。但是，这只是 eBPF 领域的一小部分。eunomia-bpf 社区提供了一个 eBPF 和 Wasm 程序的开发框架，帮助您更容易地构建、分发和部署 eBPF 程序。eunomia-bpf 社区关注于简化 eBPF 程序的编写、分发和动态加载流程，以及探索 eBPF 和 Wasm 相结合的工具链、运行时等技术：

- Github: [https://github.com/eunomia-bpf](http://link.zhihu.com/?target=https%3A//github.com/eunomia-bpf)
- Website: <https://eunomia.dev>

我们还有其他一些项目:

- [wasm-bpf](https://github.com/eunomia-bpf/wasm-bpf): 专门为 eBPF 程序打造的 WebAssembly 库、工具链和运行时，可用于构建 Wasm 用户态交互程序
- [bpf-developer-tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial): 开源的 eBPF 开发者教程和知识库，提供了一系列小工具，帮助初学者了解 eBPF 的各种用法和技巧。
- [GPTtrace](https://github.com/eunomia-bpf/GPTtrace): 使用 ChatGPT 和自然语言生成 eBPF 程序和跟踪的工具:

通过 GitHub 模板和 Codespaces，eBPF 的开发已经变得更加高效、简单和可靠。您无需再浪费时间和精力去配置环境、选择编程语言或者解决兼容性问题，而是可以专注于创造出更加高效和优秀的 eBPF 程序。感谢 eBPF 社区和 GitHub 提供这些强大的工具和支持，让我们能够更加轻松地开发 eBPF 项目，并推动这个新兴技术的发展和应用。我们相信，随着越来越多的开发者加入到 eBPF 社区中来，我们可以共同构建更加智能、高效和可靠的网络和云原生应用生态系统。如果您有任何关于 eBPF 开发的问题或建议，请随时联系我们，我们非常愿意与您交流和分享经验。
