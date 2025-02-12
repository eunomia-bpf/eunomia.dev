# About bpftime - bpftime ideas

Some basic information about the bpftime project:

## The project original and the eunomia-bpf community

1. The project original and the eunomia-bpf community

I'm maintaining a small open source organization call eunomia-bpf. We
have contributors from multiple companies and universities, including
engineers from Alibaba, Tencent some students and teachers from
multiple universities. We are trying to make eBPF easier to use, and
exploring new tool chains, runtimes for eBPF. We are also exploring
the possibilities to combining eBPF with other technologies like
WebAssembly, GPT-4, etc. I'm the founder of this organization, and the
main contributor of the bpftime project.

We are just open source organization and not a company, so we don't
have any commercial interests. The PLCT
lab(<https://plctlab.github.io/>) from Institute of Software Chinese
Academy of Sciences(<http://english.is.cas.cn/>) gives us a sponsorship
for our research and development, and help us employ
interns and students (about 1-3 people) to work on the community
projects. They don't have any copyright requirements, and My friends
and I can decide the roadmap and direction of these projects. We need
to do open-reports in public, write blog posts and tutorials, produce
better and useful open source software, make contributions to the
upstream projects, to make the community well-known and enhancing
reputations.

I also received some donations from some companies and universities
using the projects.

For the bpftime project, Two of the interns from the PLCT lab start
working on it from May this year. It was started as an open-source
project. The original goal is to make a general purpose, user space
eBPF runtime to enable 10x faster uprobe tracing, and also an
unprivileged, configurable and portable full-feature eBPF runtime,
which can run complex eBPF applications like bpftrace instead of just
some raw eBPF byte codes (like the ubpf). We are also actively
exploring more possible use cases for the bpftime. One of my friends
from USCS joined in September, currently we are also trying to make
the project into a paper with her supervisor Andrew R.
Quinn(<https://arquinn.github.io/>), for the tracing enhancement in
uprobe and unprivileged eBPF.

I'm very interested in the possible opportunity to seek more use cases
and research topics for the bpftime project, especially in the
networking area. Just like the DPDK eBPF[1], I think bpftime can also
make some changes to user space networking libraries, have a better
performance and more flexible features.

I have written a blog post about comparing other user space eBPF
runtime and bpftime, and some use cases for user space eBPF. You can
find it here: <https://eunomia.dev/blogs/userspace-ebpf/>

Some basic performance Benchmarking comparing with other user space
eBPF runtime and Wasm runtime:
<https://github.com/eunomia-bpf/bpf-benchmark>

[1]: <https://www.dpdk.org/wp-content/uploads/sites/35/2018/10/pm-07-DPDK-BPFu6.pdf>

## The possible optimization for LLVM JIT compiler

This is part of our roadmap, we will definitely do more optimization
for the bpftime runtime in the future. Even with default optimization
from LLVM, bpftime is already one of the fastest userspace eBPF
runtime comparing to ubpf and rbpf. We haven't compare it with kernel
eBPF, but I think maybe the userspace eBPF LLVM jit could benefit
from:

- kernel BPF JIT has no support for Vector Extensions like AVX, but LLVM has.
- For each architecture, kernel eBPF needs to implement its own JIT.
This will limit the optimization for each architecture, while LLVM has
a more general and powerful JIT backend, enalbing more optimization on
LLVM IR.
- It's hard to apply more and new optimization for kernel eBPF jit,
it's less configurable, any code change will need a kernel version
update in production. But for LLVM in user space, we can easily apply
new optimization and features by adding new passes or do more specific
optimization by change the compiler flags.

For current process, we are focusing on providing more features for
the bpftime runtime, make it compatible with kernel eBPF in maps,
toolchains, attach methods, verification, and helpers. The kernel
features we supported has enabled us to run applications like
bpftrace, bcc tools, observability agents and ebpf-exporters directly
in userspace without modification. However, there is still some work
to do (about a few weeks) to make it transparent to the user space
applications, and run more complex eBPF applications.

The optimization for the LLVM backend is also a must for us, maybe we
can be faster than kernel eBPF in some cases. Also, perhaps we can do
AOT compilation for the eBPF byte code, and make it more portable and
faster, without the requirement of LLVM runtime in deployment. With
AOT compilation, we can produce a nearly native speed binary, with no
runtime overhead but ensure safety, and can be easily deployed in
embedded devices. The user space eBPF is Turing complete and can
support directly function call, so it's possible to implement complex
logic in eBPF. I think the AOT compilation feature can be added easily
in one or two weeks.
