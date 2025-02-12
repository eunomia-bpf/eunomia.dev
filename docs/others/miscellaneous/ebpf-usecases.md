# Categorization of eBPF Hooks and Use Cases

#### 1. **Networking**

- **XDP Hook**: XDP allows high-performance network packet processing, such as dropping malicious packets before they reach the network stack for DDoS mitigation.
- **TC Hook**: The TC hook enables efficient traffic control by redirecting, filtering, or shaping network traffic directly at the kernel’s Traffic Control layer.
- **socket_filter Hook**: This hook allows eBPF programs to filter packets at the socket layer, enabling custom packet filtering and processing for specific applications.
    - BPF_PROG_TYPE_SOCKET_FILTER
- **Socket Lookup Hook**: This hook allows selecting the target socket for incoming network packets, improving scalability and steering traffic based on the packet’s IP or port.
- **Socket Reuseport Hook**: Facilitates efficient load balancing by selecting a socket from a set of reuseport sockets based on NUMA locality or connection state.
- **Cgroup Ingress/Egress Hook**: Allows for container-specific traffic control, such as limiting the outgoing bandwidth for a container in Kubernetes environments.
- **ULP Hook (SK_MSG)**: This hook allows eBPF programs to enforce payload-level policies for sockets, such as managing encrypted traffic with kTLS.
- **TCP Congestion Control Hook**: Registers eBPF-based TCP congestion control algorithms, allowing for custom network congestion management in production environments.
- **Socket Operations Hook**: eBPF programs can dynamically adjust socket options like TCP timeouts and header options based on peer address and connection state.

#### 2. **Profiling**

- **Perf Events Hook**: These hooks attach eBPF programs to hardware or software performance counters, enabling continuous profiling of CPU or memory usage in production environments with minimal overhead.

#### 3. **Tracing**

- **Tracepoints Hook**: Static hooks embedded in the kernel code allow eBPF to trace kernel events like system calls, such as tracing file open operations to monitor access patterns.
- **Kprobes Hook**: Dynamic hooks that can trace almost any function within the kernel, useful for monitoring critical operations like function entry and exit in real-time. It could also trace by address.
- **fentry/fexit**: These hooks allow tracing of function entry and exit points, enabling detailed performance analysis and debugging of kernel functions.
- **Uprobes Hook**: Attaches to user-space application functions, allowing eBPF to trace behaviors like user input in applications, such as monitoring readline function calls in `bash`.
- **USDT Hook**: User-level static tracepoints embedded in applications, enabling eBPF to trace custom events like function calls in user-space applications.

#### 4. **Security**

- **LSM (Linux Security Modules) Hook**: eBPF integrates with LSM hooks to dynamically enforce security policies, like applying custom memory access restrictions in real time.
- **Seccomp Hook**: Extends seccomp with eBPF filters to restrict system calls based on application behavior, such as dynamically limiting the system calls available to containerized apps. https://lwn.net/Articles/857228/ https://arxiv.org/abs/2302.10366

#### 5. **Storage and File**

- **XRP (eXpress Read Path) Hook**: eBPF programs attached to the NVMe driver layer bypass the traditional storage stack to accelerate read operations directly, enhancing NVMe performance.
- **eBPF for fuse**: eBPF programs can intercept and modify file system operations in FUSE, such as implementing custom file access policies or caching strategies. https://lwn.net/Articles/915717/ https://lpc.events/event/16/contributions/1339/attachments/945/1861/LPC2022%20Fuse-bpf.pdf

#### 6. **Memory**

- **Memory Management Hook**: eBPF can modify memory handling, such as dynamically adjusting the page reclaim policy under high memory pressure to optimize memory usage. https://lpc.events/event/18/contributions/1932/ https://arxiv.org/pdf/2409.11220
- **OOM (Out-of-Memory) Handling Hook**: eBPF programs can intercept and handle OOM events, such as prioritizing memory allocation for critical applications or processes. https://lwn.net/Articles/941614/

#### 7. **Scheduling**

- **SCHED-EXT Hook**: eBPF allows the full replacement of the kernel’s scheduler, enabling custom scheduling logic such as implementing a user-defined task management strategy. https://lpc.events/event/18/sessions/192/

#### 8. **Device Drivers**

- **HID-BPF Hook**: Enables dynamic modification of HID device drivers via eBPF, allowing custom input event filtering for devices like the Steamdeck. https://docs.kernel.org/hid/hid-bpf.html

#### 9. **Testing**

Note some Testing and Runtime features are hooks, some are not.

- **Fault Injection**: eBPF allows simulation of system faults like packet loss or memory errors to test the robustness of applications in adverse conditions, such as introducing artificial packet loss to test network resilience. https://docs.ebpf.io/linux/helper-function/bpf_override_return/

#### 10. **Runtime features**

- **Freplace**: Allows eBPF programs to replace another eBPF program at runtime, enabling dynamic program updates. Used in libxdp to allow more than one program to be attached to the same hook. https://github.com/xdp-project/xdp-tools
- **iterators**: Allows eBPF programs to iterate over kernel data structures or eBPF maps. https://docs.kernel.org/bpf/bpf_iterators.html
- **timer**: Allows eBPF programs to set timers for delayed execution, such as scheduling periodic tasks or timeouts. https://lwn.net/Articles/862136/

## The comparison of eBPF program types, attach types and event types

https://docs.kernel.org/bpf/libbpf/program_types.html

It has a table that shows the comparison of eBPF program types, attach types and events defined in the elf, such as `cgroup/sendmsg6` or `perf_event`

### Definitions of Core Requirements

1. **Performance:** Extensions must operate efficiently to maintain or enhance system performance without introducing significant overhead.
2. **Security:** Extensions should protect against vulnerabilities and ensure the integrity and confidentiality of the system.
3. **Expressiveness:** Extensions need the capability to define and implement complex, detailed, and customizable logic or policies.
4. **Dynamic:** Extensions must be able to adjust to varying conditions, workloads, or requirements in real-time, including attaching to or modifying system behavior dynamically.
5. **Control:** Extensions should provide mechanisms for external sources to manage, adjust, and fine-tune behaviors, parameters, or policies.
6. **Integration Scope:** Extensions must appropriately interact with system components, ranging from minimal interaction for lightweight subsystems to deep integration for more complex systems.

### Core Requirements for eBPF Extensions

#### 1. **Performance**
Extensions must operate efficiently to maintain or enhance system performance without introducing significant overhead.

#### 2. **Security**
Extensions should protect against vulnerabilities and ensure the integrity and confidentiality of the system.

#### 3. **Expressiveness**
Extensions need the capability to define and implement complex, detailed, and customizable logic or policies.

#### 4. **Dynamic**
Extensions must be able to adjust to varying conditions, workloads, or requirements in real-time, including attaching to or modifying system behavior dynamically.

#### 5. **Control**
Extensions should provide mechanisms for external sources to manage, adjust, and fine-tune behaviors, parameters, or policies.

#### 6. **Integration Scope**
Extensions must appropriately interact with system components, ranging from minimal interaction for lightweight subsystems to deep integration for more complex systems.

---

### Summary of Core Requirements Importance

| **Category**                                | **Performance** | **Security** | **Expressiveness** | **Dynamic** | **Control** | **Integration Scope** |
|---------------------------------------------|-----------------|--------------|--------------------|-------------|-------------|-----------------------|
| **1. Networking**                           | High            | Medium       | High               | High        | High        | Minimal to Moderate   |
| **2. Profiling**                            | High            | Medium       | Medium             | Medium      | Medium      | Minimal               |
| **3. Tracing**                              | Medium          | Medium       | High               | High        | High        | Moderate               |
| **4. Security**                             | Medium          | High         | Medium             | High        | High        | Moderate to Deep      |
| **5. Storage and File**                     | High            | Medium       | High               | Medium      | Medium      | Moderate               |
| **6. Memory**                               | High            | Medium       | Medium             | High        | Medium      | Moderate               |
| **7. Scheduling**                           | High            | Medium       | High               | High        | High        | Deep                  |
| **8. Device Drivers**                       | High            | Medium       | High               | High        | Medium      | Deep                  |
| **9. Testing**                              | Medium          | Medium       | Medium             | High        | High        | Minimal to Moderate   |
| **10. Runtime Features**                    | High            | Medium       | High               | High        | High        | Moderate to Deep      |

### Key Insights

1. **Performance** is crucial for **Networking**, **Profiling**, **High-Speed Data Processing**, **Scheduling**, and **Device Drivers**, where efficiency and low overhead are essential.
2. **Security** is paramount in **Security**, **Access Control**, and **Patch Fixing**, focusing on protecting against threats and vulnerabilities.
3. **Expressiveness** allows for defining complex policies and logic, important across **Networking**, **Tracing**, **Storage and File**, and **Scheduling**.
4. **Dynamic** capability is essential for **Networking**, **Tracing**, **Security**, **Memory**, and **Runtime Features**, enabling real-time adaptation.
5. **Control** mechanisms are vital for **Networking**, **Tracing**, **Security**, **Testing**, and **Runtime Features**, facilitating dynamic management and adjustments.
6. **Integration Scope** varies significantly, with **Device Drivers** and **Scheduling** requiring deep integration, while **Networking** and **Profiling** often need minimal to moderate interaction.

---

### Additional Considerations

- **Integration Scope** helps determine the level of interaction required between extensions and system components, ensuring that extensions are appropriately designed for their target use cases.
- Balancing **Performance** and **Security** is often critical, especially in high-impact areas like **Networking** and **Scheduling**.
- **Expressiveness** and **Dynamic** capabilities enable extensions to handle complex and evolving scenarios, making them more versatile and powerful.
- **Control** mechanisms ensure that extensions can be managed effectively, allowing for fine-tuning and adjustments as system requirements change.

By adhering to these core requirements, designers and developers can create robust, efficient, and secure eBPF extensions tailored to their specific use cases.