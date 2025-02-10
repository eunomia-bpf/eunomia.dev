---
date: 2024-04-11
---
# Implementing an Inline Hook in C in 5 minutes

One of the fascinating aspects of programming comes when we try to alter the behavior of a program while it is running.

In this tutorial, we shed light on one method that can make this possible - an "Inline Hook". We will delve into how you can manipulate the execution flow of a program in the C programming language. By implementing an Inline Hook, we aim to divert the program's execution flow to our function, then returning it back to the normal flow.
<!-- more -->

## What is an Inline Hook?

An Inline hook is a technique that inserts a piece of code into a running program, altering its control flow. In practice, this is achieved by replacing the first few instructions of a function with a jump to our inserted code (usually another function), which upon completion will jump back, continuing the execution of the original function. Frida is a popular tool that uses this technique to inject code into a running process. It is used for dynamic instrumentation, debugging, and reverse engineering.

In our userspace eBPF runtime bpftime (<https://github.com/eunomia-bpf/bpftime>), we use inline hooking to implement the `uprobe` feature. bpftime is an userspace eBPF runtime that allows existing eBPF applications to operate in unprivileged userspace using the same libraries and toolchains. It offers Uprobe and Syscall tracepoints for eBPF, with significant performance improvements over kernel uprobe and without requiring manual code instrumentation or process restarts.

## Inline Hook Implementation

The Inline hook implementation primarily follows five crucial steps:

1. Identifying the memory address of the function to be hooked.
2. Backing up the initial instructions of the target function that will be overwritten,
3. Writing a jump instruction at the beginning of the target function in the hooked process's memory,
4. Creating the hook function, which will replace the original one,
5. Altering the memory permissions to enable the changes, and restoring them once modifications are complete.

On a side note, Inline Hooking could be limited by modern compiler optimization and certain memory protection procedures such as Data Execution Prevention (DEP) and Address Space Layout Randomization (ASLR).

## Inline Hooking Example: how to use it

To make this more digestible, we will use an example scenario. In this example, we will hook a simple function `my_function`. This code is in `main.c` and it initially prints "Hello, world!". But after applying our hook, it prints "Hello from hook!" instead.

```c
// This is the original function to hook.
void my_function()
{
    printf("Hello, world!\n");
}
```

Next, we create a hooking function `my_hook_function` in `hook.c`. This function will replace `my_function` and is designed to print "Hello from hook!"

```c
// This is the hook function.
void my_hook_function()
{
    printf("Hello from hook!\n");
}
```

The `inline_hook` function is the most critical part of our application. It uses `mprotect` to change the memory permissions of the target function, making it writable. It then replaces the first few instructions of `my_function` with a jump instruction to `my_hook_function`. The original bytes are saved for future restoration.

On the main function, we start by calling `my_function`, enabling the `inline_hook`, calling `my_function` again (which now executes `my_hook_function`), then removing the hook and calling `my_function` another time to see that it prints the original "Hello, world!" string.

```c
int main()
{
    my_function();
    
    // Enabling the hook.
    inline_hook(my_function, my_hook_function);
    
    // Now calling the function will actually call the hook function.
    my_function();
    
    // Removing the hook
    remove_hook(my_function);
    
    // Now calling the function will call the original function.
    my_function();

    return 0;
}
```

After compiling and running the main function, we can observe the output.

```console
$ make
$ ./maps
Hello, world!
Hello from hook!
Hello, world!
```

You can find the complete example in the following repository: [https://github.com/eunomia-bpf/inline-hook-demo](https://github.com/eunomia-bpf/inline-hook-demo)

## Implementation a inline hook

Let's take a look at the implementation of the `inline_hook` function. This is a very basic implementation that works on x86_64, ARM64, and ARM32. It is not a complete implementation, but it should be enough to get you started.

```c
#include <sys/mman.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#define SIZE_ORIG_BYTES 16
static void inline_hook_replace_inst(void *orig_func, void *hook_func) {
        // Write a jump instruction at the start of the original function.
    *((unsigned char *)orig_func + 0) = 0xE9; // JMP instruction
    *((void **)((unsigned char *)orig_func + 1)) =
        (unsigned char *)hook_func - (unsigned char *)orig_func - 5;
}
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SIZE_ORIG_BYTES 32
static void inline_hook_replace_inst(void *orig_func, void *hook_func) {
    int offset = ((intptr_t)hook_func - (intptr_t)orig_func) / 4;
    if (offset < -0x2000000 || offset > 0x1ffffff) {
        printf("Offset %d out of range!\n", offset);
        exit(1);
    }
    uint32_t branch_instruction = 0x14000000 | (offset & 0x03ffffff);
    *((uint32_t*)orig_func) = branch_instruction;
}
#elif defined(__arm__) || defined(_M_ARM)
#define SIZE_ORIG_BYTES 20
static void inline_hook_replace_inst(void *orig_func, void *hook_func) {
    // Construct a branch instruction to the hook function.
    // The instruction for a branch in ARM is 0xEA000000 | ((<offset> / 4) & 0x00FFFFFF)
    // The offset needs to be divided by 4 because the PC advances by 4 bytes each step in ARM
    int offset = ((intptr_t)hook_func - (intptr_t)orig_func - 8) / 4;
    int branch_instruction = 0xEA000000 | (offset & 0x00FFFFFF);

    // Write the branch instruction to the start of the original function.
    *(int *)orig_func = branch_instruction;
}
#else
#error "Unsupported architecture"
#endif

void *get_page_addr(void *addr)
{
    return (void *)((uintptr_t)addr & ~(getpagesize() - 1));
}

unsigned char orig_bytes[SIZE_ORIG_BYTES];

void inline_hook(void *orig_func, void *hook_func)
{
    // Store the original bytes of the function.
    memcpy(orig_bytes, orig_func, SIZE_ORIG_BYTES);

    // Make the memory page writable.
    mprotect(get_page_addr(orig_func), getpagesize(),
         PROT_READ | PROT_WRITE | PROT_EXEC);

    inline_hook_replace_inst(orig_func, hook_func);

    // Make the memory page executable only.
    mprotect(get_page_addr(orig_func), getpagesize(),
         PROT_READ | PROT_EXEC);
}

void remove_hook(void *orig_func)
{
    // Make the memory page writable.
    mprotect(get_page_addr(orig_func), getpagesize(),
         PROT_READ | PROT_WRITE | PROT_EXEC);

    // Restore the original bytes of the function.
    memcpy(orig_func, orig_bytes, SIZE_ORIG_BYTES);

    // Make the memory page executable only.
    mprotect(get_page_addr(orig_func), getpagesize(),
         PROT_READ | PROT_EXEC);
}
```

We start by saving the original bytes of the target function in the `orig_bytes` array. We then make the memory page writable using `mprotect`. Next, we replace the first few instructions of the target function with a jump instruction to the hook function. Finally, we restore the memory page's permissions to their original state. `get_page_addr` computes the page-aligned address. `inline_hook` sets up the hook by storing original bytes and modifying instructions. `remove_hook` reverses the changes.

The hook installation differs based on the processor architecture.

On x86_64, we replace the beginning of the target function with a `JMP` instruction that redirects to our hook function.

```c
#define SIZE_ORIG_BYTES 16
static void inline_hook_replace_inst(void *orig_func, void *hook_func) {
    // Write a jump instruction at the start of the original function.
    *((unsigned char *)orig_func + 0) = 0xE9; // JMP instruction
    *((void **)((unsigned char *)orig_func + 1)) =
        (unsigned char *)hook_func - (unsigned char *)orig_func - 5;
}
```

Note that in ARM32, the Program Counter (PC) is usually 2 instructions ahead, which is why we subtract 8 (2 instructions * 4 bytes/instruction) when calculating the offset. This might differ between different ARM versions or modes (Thumb vs ARM, etc.) so please adjust accordingly to your target's specifics.

Also, you need to increase the SIZE_ORIG_BYTES from 16 to 20 because the minimal branch instruction in ARM is 4 bytes and you're going to replace 5 instructions. This is needed because the branch instruction uses a relative offset and you cannot be sure how far your hook function will be. If your function and hook are within 32MB of each other, you could only replace the first 4 bytes with a branch and wouldn't need to touch the rest.

```c
#define SIZE_ORIG_BYTES 20
static void inline_hook_replace_inst(void *orig_func, void *hook_func) {
    int offset = ((intptr_t)hook_func - (intptr_t)orig_func - 8) / 4;
    int branch_instruction = 0xEA000000 | (offset & 0x00FFFFFF);
    *(int *)orig_func = branch_instruction;
}
```

Similar to ARM32, ARM64 uses the ARM instruction set. However, there are differences and specifics to consider for ARM64. For example, the encoding of the branch instruction is different and because of the larger address space, you have to create a trampoline for larger offsets that can't be reached by a single branch instruction. The trampoline should be close to the original function so it can be reached by a branch instruction and from there, it will load the full 64 bit address of the hook function.

```c
#define SIZE_ORIG_BYTES 32
static void inline_hook_replace_inst(void *orig_func, void *hook_func) {
    int offset = ((intptr_t)hook_func - (intptr_t)orig_func) / 4;
    // Check if the offset is within the allowable range for a branch instruction.
    if (offset < -0x2000000 || offset > 0x1ffffff) {
        printf("Offset %d out of range!\n", offset);
        exit(1);
    }
    // Construct and write the branch instruction.
    uint32_t branch_instruction = 0x14000000 | (offset & 0x03ffffff);
    *((uint32_t*)orig_func) = branch_instruction;
}
```

You can find the complete example in the following repository: [https://github.com/eunomia-bpf/inline-hook-demo](https://github.com/eunomia-bpf/inline-hook-demo)

## Limitations

## Understanding the Limitations of Inline Hooking

Inline Hooking, while a powerful technique for intercepting and modifying function calls in software, has several inherent limitations, particularly in the context of modern operating systems and programming environments. Here, we delve into these limitations in more detail to provide a clearer understanding of the challenges and implications involved. The demostration code is very simple and cannot be used in production.

### 1. **Operating System Security Mechanisms**

Modern operating systems deploy a variety of security mechanisms to prevent malicious or unintended modifications to executing code:

- **Data Execution Prevention (DEP)**: DEP is designed to prevent code from being run from data segments of a process, such as the stack or heap. Inline hooking often requires executing code that has been written to these segments, which can be blocked by DEP.

- **Address Space Layout Randomization (ASLR)**: ASLR randomizes the memory addresses used by system and application files. This complicates the process of inline hooking since the exact address of the target function may change every time the application or system is restarted.

- **Code Signing and Integrity Checks**: Some operating systems and applications implement code signing and integrity checks. These mechanisms can detect modifications to code, including inline hooks, and may prevent the modified application from executing or flag it as malicious.

### 2. **Compiler Optimizations**

Modern compilers employ various optimizations that can interfere with inline hooking:

- **Function Inlining**: Compilers may inline functions, which means the function's code is directly inserted into each place it is called, rather than being kept as a separate function. This can eliminate the consistent function entry point that inline hooks rely on.
- **Instruction Reordering and Optimizations**: Compilers might reorder instructions or optimize the function's structure in a way that doesn't align well with the assumptions made during the setup of an inline hook, potentially leading to crashes or undefined behavior.

### 3. **Multi-threading and Concurrent Execution**

- **Thread Safety**: In multi-threaded applications, ensuring that the hook is correctly applied without interrupting currently executing threads can be challenging. There's a risk of race conditions where one thread might be executing the function being hooked while another is applying the hook.
- **Re-entrancy Issues**: If the hooked function or the hook itself is re-entrant (can be called simultaneously from multiple threads), it complicates the inline hooking process. Care must be taken to handle such cases properly to avoid deadlocks or inconsistent program states.

### 4. **Hardware and Architecture Specifics**

- **Instruction Set Differences**: Different processors have different instruction sets and execution models. For instance, ARM and x86 processors have significantly different ways of handling instructions, making the process of writing a universal inline hook more complex.
- **Instruction Length Variations**: The length of instructions can vary (especially in variable-length instruction sets like x86), making it difficult to determine how many bytes to overwrite safely without affecting subsequent instructions.

## Wrapping Up

Understanding inline hooking can substantially aid in areas such as software security, testing, and debugging. It provides an avenue to alter and control program behavior on-the-fly. While it is powerful, it also comes with its drawbacks, which need to be handled with care.
In conclusion, while inline hooks are powerful tools, they should be used with caution, understanding, and a good knowledge of system architecture.

I hope you enjoyed the journey exploring Inline Hooks. Happy coding!
