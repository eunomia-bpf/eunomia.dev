# Userspace eBPF Runtimes: A Deep Dive into Overview, Applications, and Performance Benchmarks

In this blog post, we will talked baout ebpf, the userspace ebpf runtimes and their use cases. We have also benchmarked the performance of userspace ebpf runtimes and compared it with Wasm and native code. We also created a new userspace eBPF runtime called [bpftime](https://github.com/eunomia-bpf/bpftime), based on LLVM for JIT.

## Introduction

- What is eBPF?
- Why is understanding userspace eBPF runtimes important?

## eBPF: from kernel to userspace

- Brief history of eBPF.
- Its significance in modern computing and network solutions.

## Userspace Runtimes and Their Role

- What is a userspace runtime?
- Introduction to specific runtimes:
  - **ubpf**: Key features and advantages.
  - **rbpf**: Understanding its unique strengths.
  - **bpftime**: A deeper look into this new runtime based on LLVM for JIT.

## Use Cases: Existing eBPF Userspace Applications

- How eBPF is currently utilized in various applications.
- The benefits of using eBPF in these applications.

### Benchmark Showdown

- The methodology behind the benchmark tests.
- Comparison of performance:
  - eBPF Runtimes vs. Wasm.
  - eBPF Runtimes vs. Native Code.
  
### The Future of eBPF and Its Userspace Runtimes

- Predictions about the trajectory of eBPF.
- How developers can potentially leverage eBPF in future projects.

### Conclusion

- Key takeaways from the exploration of userspace eBPF runtimes.
- Encouraging readers to explore the [bpftime GitHub repository](https://github.com/eunomia-bpf/bpftime) and contribute.

### References

- Citing sources and further reading materials for those interested.
