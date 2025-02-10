---
date: 2023-07-11
---

# Simplifying eBPF Development: GitHub Templates and Codespaces for Online Compilation and Execution

Embarking on the eBPF journey can feel daunting, especially when confronted with setting up the perfect environment or making the ideal language choice. But what if there was a streamlined way to immerse yourself in eBPF without the initial hurdles? Look no further! This guide unveils the magic of GitHub templates combined with GitHub Codespaces, empowering you to seamlessly initiate, compile, and run eBPF projects online. Dive in, click once, and turbocharge your eBPF expedition!
<!-- more -->

## eBPF: Unveiling the Linux Kernel's Next Frontier

At the heart of the Linux kernel lies eBPF (Extended Berkeley Packet Filter) — a revolutionary technology designed to dynamically inject and execute bite-sized programs, enriching the kernel's capabilities in realms like networking, security, and performance monitoring. Whether it's TCP/IP, UDP, or ICMP packets, eBPF manages them all, flexing its muscles in both user-space and kernel-space programming. 

But here's the catch: initiating an eBPF project often comes wrapped in complexities. From deep kernel insights to juggling diverse toolchains, and not to mention the pressing concerns of performance and security, developers can easily find themselves in a maze. But with the tools and insights we're about to share, your eBPF journey is about to get a whole lot smoother.

## How to Swiftly Set Up an Environment and Choose a Programming Language?

When looking to create an eBPF project, are you puzzled about setting up the environment or selecting a programming language? Worry not! We've prepared a set of GitHub templates for you, enabling you to swiftly begin a fresh eBPF project. Simply click the `Use this template` button on GitHub to start.

## GitHub Templates: Fast-Track Your eBPF Project

Setting up the environment and basic configurations are essential yet tedious when you start an eBPF project. To save you time and effort, we've crafted a series of GitHub templates catering to different programming languages and frameworks to meet your varied needs. These templates let you quickly kick off a brand-new eBPF project without building everything from scratch.

- [libbpf-starter-template](https://github.com/eunomia-bpf/libbpf-starter-template): An eBPF project template based on the C language and the libbpf framework.
- [cilium-ebpf-starter-template](https://github.com/eunomia-bpf/cilium-ebpf-starter-template): An eBPF project template based on the C language and the cilium/ebpf framework.
- [libbpf-rs-starter-template](https://github.com/eunomia-bpf/libbpf-rs-starter-template): An eBPF project template based on the Rust language and the libbpf-rs framework.
- [eunomia-template](https://github.com/eunomia-bpf/eunomia-template): An eBPF project template based on the C language and the eunomia-bpf framework.

These starter templates come with the following features:

- A Makefile that allows you to build the project with one click.
- A Dockerfile designed for automatically creating a containerized environment for your eBPF project and publishing it to GitHub Packages.
- GitHub Actions for automating the build, test, and release processes.
- All dependencies required for eBPF development.

By setting up an existing repository as a template, both you and others can quickly generate new repositories with the same foundational structure. This eliminates the tedious process of manual creation and configuration. GitHub templates offer developers a simple way to kickstart a new project while quickly leveraging best practices and experiences shared by other developers.

Once your repository is set as a template, other users can create a new one by:

1. Opening the template repository page.
2. Clicking on the "Use this template" button.
3. Entering the new repository's name and description, and choosing an organization or personal account for its creation.
4. Clicking on the "Create repository from template" button to finalize.

![Image Link](https://picx.zhimg.com/80/v2-9147b573ee3df2d0f955fc62fb81128b_1440w.webp?source=d16d100b)

(Note: If auto-publishing images to GitHub Packages fails in the CI process, you may need to configure action permissions in the repository's Settings. Refer to the later sections for guidance.)

With GitHub templates, you can quickly launch a new project, fast-track your understanding of best practices shared by other developers, and focus more on your project's core functionalities and logic, thereby improving development efficiency and code quality. In a GitHub template, you can choose to develop eBPF programs using the C language, Rust, or the eunomia-bpf framework. Additionally, you can utilize GitHub Actions for automating the building, testing, and releasing of eBPF binary and container images. For more detailed information, refer to the official documentation: [Official Documentation Link](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-template-repository)

## **GitHub Codespaces: Compile eBPF Programs in the Cloud**

GitHub Codespaces is a cloud-based development environment that offers rapid and flexible capabilities to compile and run eBPF programs. No more local software installations and configurations; you can easily build and debug eBPF programs using GitHub's primary services.

Within Codespaces, each code repository has a corresponding Docker container containing all the tools and dependencies needed for your project. You can connect to Codespaces via a browser, Visual Studio Code, or JetBrains Gateway, accessing a fully customizable cloud-based development environment.

Developing eBPF programs with GitHub Codespaces is straightforward:

1. Open your eBPF project repository.
2. Click on the "Code" button and then select the "Open with Codespaces" option.

![Open with Codespaces](https://picx.zhimg.com/80/v2-c97afd28cc88ad73836eb4561267021c_1440w.webp)

3. GitHub will create a new Codespace for you, which will be equipped with all the required tools and dependencies for your project. This may take a few minutes, depending on your network speed and the repository size.
4. Once your Codespace is ready, you can open the terminal, navigate to your project directory, and begin compiling and running your eBPF program.

![codespace](https://pic1.zhimg.com/80/v2-8b37f9241c284ac494555149272d1e57_1440w.webp?source=d16d100b)

Using GitHub Codespaces, you can bypass environmental setup, version compatibility issues, and always employ the latest development tools and frameworks. There's no need to worry about system hardware limitations, performance issues, or being confined to a particular location or device. With Codespaces, you can solely focus on coding and the project's core functionalities, enhancing development efficiency and code quality, ensuring faster and more efficient project progression. For more details, refer to: [GitHub Codespaces Link](https://github.com/codespaces)
## **Run eBPF Program with Docker in One Step**

Write code in codespace, and upon submission, Github Actions will compile and automatically publish a container image. To use the automated image publishing feature, configure actions permissions in the repository's Settings:

![actions](https://pic1.zhimg.com/80/v2-2e0f9fc6aa0d1aee4231963432105626_1440w.webp)

Next, you can run the eBPF program anywhere with docker in one step, like this:

`sudo docker run --rm -it --privileged ghcr.io/eunomia-bpf/libbpf-rs-template:latest`

![dockerdocker](https://pic1.zhimg.com/80/v2-ede596564dc3a701889ed161dcda9eb5_1440w.webp?source=d16d100b)

## **Conclusion**

In the development of eBPF projects, we introduced how to quickly set up the environment, choose programming languages, and how to use the GitHub template to start projects. This saves a lot of time and effort for developers, allowing them to focus on creating core features and business logic. However, this is just a small part of the eBPF field. The eunomia-bpf community provides a development framework for eBPF and Wasm programs to make it easier for you to build, distribute, and deploy eBPF programs. Eunomia-bpf is committed to simplifying the writing, distribution, and dynamic loading of eBPF programs, as well as exploring the combination of eBPF and Wasm toolchains and runtimes:

- Github: [https://github.com/eunomia-bpf](http://link.zhihu.com/?target=https%3A//github.com/eunomia-bpf)
- Website: <https://eunomia.dev>

We also have some other projects:

- [wasm-bpf](https://github.com/eunomia-bpf/wasm-bpf): A WebAssembly library, toolchain, and runtime designed specifically for eBPF programs, allowing for the building of Wasm user-space interactive programs.
- [bpf-developer-tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial): An open-source eBPF developer tutorial and knowledge base, offering a set of utilities to help beginners understand the various uses and tricks of eBPF.
- [GPTtrace](https://github.com/eunomia-bpf/GPTtrace): Tools for generating eBPF programs and traces using ChatGPT and natural language.

With the GitHub template and Codespaces, eBPF development has become more efficient, simpler, and reliable. You no longer need to waste time and energy configuring environments, choosing programming languages, or resolving compatibility issues. Instead, you can focus on creating more efficient and superior eBPF programs. We thank the eBPF community and GitHub for providing these powerful tools and support, enabling us to develop eBPF projects more easily and promote the development and application of this emerging technology. We believe that as more developers join the eBPF community, together we can build a smarter, more efficient, and reliable network and cloud-native application ecosystem. If you have any questions or suggestions about eBPF development, please feel free to contact us. We are always eager to communicate and share experiences.