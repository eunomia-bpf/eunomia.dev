# Uprobe and Syscall Tracing with bpftime

## Uprobe Tracing with bpftime

Uprobes(User Probe) allow you to attach dynamic probes to user-space functions, enabling you to monitor and trace specific user-level functions or methods in your applications. bpftime can be utilized within BPF programs attached to uprobes to capture timestamps for different events related to that function.\
Here's how you can use bpftime for uprobe tracing:

### Install BCC

Make sure you have [BCC](https://github.com/iovisor/bcc) installed on your system. Follow the installation instructions for your Linux distribution [here](https://github.com/iovisor/bcc/blob/master/INSTALL.md).

### Use bpftime for Uprobe Tracing

` sudo bpftime 'uretprobe:/path/to/executable:function' `

Replace `/path/to/executable` with the path to the executable binary you want to trace, and replace function with the name of the function you want to trace.

### Example

`sudo bpftime 'uretprobe:/usr/bin/ls:main'`

This command attaches a BPF program to the main function of the ls command and traces the time taken by the function.

## Syscall Tracing with bpftime

Syscall tracing involves monitoring system calls made by processes running on your system. By adding bpftime to your BPF programs, you can capture timestamps for various syscall events, such as entry and exit.\
Here's how you can use bpftime for syscall tracing:

### Use bpftime for Syscall Tracing

`sudo bpftime 'tracepoint:raw_syscalls:sys_enter'`

This command traces the time taken by system calls when they enter.

### Filtering by Specific Syscall

You can filter by a specific syscall using the -F option:

`sudo bpftime -F read 'tracepoint:syscalls:sys_enter_read'`

This command traces the time taken by the read system call.

### Output Format:
By default, bpftime outputs a table with the following columns:

TIME(s):$~$ Time in seconds.\
PID:$~$ Process ID.\
COMM:$~$ Process name.\
TYPE:$~$ Event type (syscall or uretprobe).\
EVENT:$~$ Event name.\
DURATION(ms):$~$ Duration in milliseconds.

### Example BPF Program (Using bpftrace):
Here's a simple example of a BPF program written in bpftrace that captures timestamps for open() syscalls:

```c
tracepoint:syscalls:sys_enter_open
{
    @start[pid] = bpftime();
}

tracepoint:syscalls:sys_exit_open
/ @start[pid] /
{
    printf("PID %d: open() took %lld nanoseconds\n", pid, bpftime() - @start[pid]);
}
```
This program attaches to the open() syscall entry and exit points, captures timestamps using bpftime(), and calculates the time taken for each open() syscall.

Remember, bpftime provides high-resolution timestamps and is suitable for analyzing microsecond-level timing information. Ensure that your system supports BPF and has the necessary permissions to load and attach BPF programs.

Refer to the documentation of [bpftrace](https://github.com/bpftrace/bpftrace/blob/master/man/adoc/bpftrace.adoc), bcc, or other BPF-related tools for more advanced usage and examples of bpftime in action. Additionally, explore the eBPF documentation and kernel sources for detailed information on eBPF features and capabilities.

### Additional Options

#### Interval and Count

You can use the -i option to set the interval (in seconds) and the -c option to set the count:

`sudo bpftime -i 1 -c 10 'tracepoint:syscalls:sys_enter'`

This command traces the time taken by system calls every second for a total of 10 samples.

### Output to File

You can redirect the output to a file using standard shell redirection:

`sudo bpftime 'tracepoint:syscalls:sys_enter' > output.txt`

These are basic examples to get you started. Adjust the commands based on your specific requirements and the events you want to trace. Refer to the [BCC documentation](https://github.com/iovisor/bcc/blob/master/docs/reference_guide.md) for more advanced usage and options.