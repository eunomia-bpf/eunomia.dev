# The design and implementation of bpftime

The hook implementation is based on binary rewriting and the underly technique is inspired by:

- Userspace function hook: [frida-gum](https://github.com/frida/frida-gum)
- Syscall hooks: [zpoline: a system call hook mechanism based on binary rewriting](https://www.usenix.org/conference/atc23/presentation/yasukata) and [pmem/syscall_intercept](https://github.com/pmem/syscall_intercept).

For more details about how to implement the inline hook, please refer to our blog: [Implementing an Inline Hook in C in 5 minutes](../../blogs/inline-hook.md) and the demo <https://github.com/eunomia-bpf/inline-hook-demo>

The injection of userspace eBPF runtime into a running program is based on ptrace and also provided by [frida-gum](https://github.com/frida/frida-gum) library.

## How the bpftime work entirely in userspace:

![How it works](bpftime.png)

bpftime runs primarily in the kernel environment and leverages the capabilities of the eBPF (Extended Berkeley Packet Filter) program. eBPF services are loaded into the kernel and run in a confined and secure environment, allowing them to perform a variety of monitoring and monitoring tasks.

Below is a simple summary of how to use bpftime with eBPF in a user environment:

- User interacts with bpftime from the Command line. They show events to track, such as phone calls or calls made to users.

- When you run bpftime, it converts user commands into instructions for loading and extending eBPF programs. It also shows the events that need to be monitored and the relevant filters.

- bpftime loads precompiled eBPF programs into the kernel. These programs contain logic to capture time and calculate the timing of events.

- eBPF services are linked to specific events, such as phone calls or user calls. When these events occur, the relevant eBPF service is triggered.

- eBPF programs use BPF support functions (such as `bpf_ktime_get_ns()`) to capture timestamps. These time logs represent the start and end times of tracked events.

- The eBPF program calculates the time of the event using time capture. This information is usually stored in a file shared between the kernel and user space.

- Periodically or when the bpftime command terminates, user space saves the data in the shared file.

- bpftime output trace data and output to the user via the command line interface. This output usually includes details such as process ID, process name, event type, event name, and duration.

In summary, the main concepts of timestamp capture, time calculation and event tracking occur in the kernel through eBPF programs, while bpftime follows a client-side tool that helps configure, load and retrieve data from eBPF programs. The combination of eBPF and bpftime provides a powerful tool to monitor and analyze events efficiently and cost-effectively.

## How the bpftime work with kernel eBPF:

![How it works with kernel eBPF](bpftime-kernel.png)

bpftime is used in conjunction with the eBPF (Extended Berkeley Packet Filter) program in the kernel to monitor events and measure the duration of these events in the kernel. 

Below is a summary of how bpftime interacts with the eBPF core:


- Users execute bpftime commands from command line users, specifying the events they want to monitor.

- bpftime defines user commands and settings, determines the types of events that should be monitored (such as calls, calls to users), and other options.

- bpftime loads precompiled eBPF programs into the kernel. These eBPF programs are responsible for maintaining time logs and calculating the time of specified events.

- The loaded eBPF program is added to a specific direction or location detected in the source of interest. For example, add tracepoint:raw_syscalls:sys_enter to capture the beginning of the call.

- When an event occurs in the kernel (for example, a system call), additional eBPF processes are triggered. eBPF services run in a constrained environment beyond the kernel.

- eBPF programs use BPF support functions (such as bpf_ktime_get_ns()) to capture timestamps. These time logs represent the start and end times of tracked events.

- The eBPF program calculates the time of the event using time capture. This information is usually stored in the master file or sent to the user site for further processing.

- bpftime periodically receives or polls time count and other relevant data from the eBPF program. son This can be done using communication techniques such as BPF maps or other integrated data.

- BPFtime input and output monitoring information is delivered to the user via the command line interface. The output usually includes details such as process ID, process name, event type, event name, and duration.

In essence, bpftime acts as a user interface that integrates the loading and configuration of the eBPF program into the kernel, while the actual monitoring and measuring time is done by the eBPF program itself in the main place. This collaboration allows critical events to be tracked efficiently and seamlessly.

For more details, please refer to:

- Slides: <https://eunomia.dev/bpftime/documents/userspace-ebpf-bpftime-lpc.pdf>
- arxiv: <https://arxiv.org/abs/2311.07923>
