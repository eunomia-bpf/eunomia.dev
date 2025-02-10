---
date: 2024-04-11
---

# 五分钟带你手搓一个简易的 inline hook 实现控制流劫持

编程中令人着迷的一面在于我们尝试在程序运行时改变其行为。在本教程中，我们将揭示一种可以实现这一点的方法- inline hook 。只需要数十行代码，即可用 C 语言实现一个简单的 inline hook 示例，并将其应用于一个示例程序。

我们将探讨如何在 C 编程语言中操纵程序的执行流。通过实现 inline hook ，我们的目标是将程序的执行流分流到我们的函数中，然后再返回到正常流程。您可以在以下存储库中找到完整的开源代码示例： [https://github.com/eunomia-bpf/inline-hook-demo](https://github.com/eunomia-bpf/inline-hook-demo)
<!-- more -->

## 什么是 inline hook ？

inline hook 是一种在运行的程序中插入一段代码的技术，从而改变其控制流的方法。实际上，这是通过用一个跳转到我们插入的代码（通常是另一个函数）的跳转来取代函数的最初几条指令来实现的，该代码在完成后会跳回，继续执行原始函数。Frida是一种流行的工具，它使用这种技术将代码注入到运行的进程中。它用于动态仪器化、调试和逆向工程。

在我们的用户空间 eBPF 运行时 bpftime (<https://github.com/eunomia-bpf/bpftime>)中，我们使用 inline hook 来实现`uprobe`功能。bpftime 是一个用户空间的eBPF运行时，允许现有的eBPF应用在非特权用户空间中使用相同的库和工具链。它为eBPF 提供了Uprobe和 Syscall跟踪点，并且在不需要手动代码仪器化或进程重启的情况下，具有显著的性能提升。当然，实际的 Uprobe 实现要比本文讨论的复杂得多。

## inline hook 实现

inline hook 实现主要遵循以下五个关键步骤：

1. 确定要挂钩的函数的内存地址。
2. 备份将要被覆盖的目标函数的初始指令。
3. 在挂钩进程的内存中目标函数的开头写入跳转指令。
4. 创建替代原始函数的钩子函数。
5. 更改内存权限以允许修改，并在完成修改后恢复它们。

顺便提一句， inline hook 可能会受到现代编译器优化和某些内存保护过程（如数据执行预防(DEP)和地址空间布局随机化(ASLR)）的限制。

## inline hook 示例：如何使用它

为了使这更容易理解，我们将使用一个示例场景。在这个示例中，我们将挂钩一个简单的函数`my_function`。这段代码位于`main.c`中，最初打印"Hello, world!"。但是在应用我们的钩子之后，它将打印"Hello from hook!"。

```c
// 这是要挂钩的原始函数。
void my_function()
{
    printf("Hello, world!\n");
}
```

接下来，我们在`hook.c`中创建一个钩子函数`my_hook_function`。这个函数将替换`my_function`，并设计为打印"Hello from hook!"。

```c
// 这是钩子函数。
void my_hook_function()
{
    printf("Hello from hook!\n");
}
```

`inline_hook`函数是我们应用中最关键的部分。它使用`mprotect`更改目标函数的内存权限，使其可写。然后，它使用跳转指令将`my_function`的前几条指令替换为跳转到`my_hook_function`。原始字节保存用于将来恢复。

在`main`函数中，我们首先调用`my_function`，启动`inline_hook`，再次调用`my_function`（现在执行`my_hook_function`），然后移除钩子并再次调用`my_function` 以查看它是否打印原始的"Hello, world!"字符串。

```c
int main()
{
    my_function();

    // 启用钩子。
    inline_hook(my_function, my_hook_function);

    // 现在调用该函数将会调用钩子函数。
    my_function();

    // 移除钩子
    remove_hook(my_function);

    // 现在调用函数将会调用原始函数。
    my_function();

    return 0;
}
```

编译和运行主函数后，我们可以观察输出结果。

```console
$ make
$ ./maps
Hello, world!
Hello from hook!
Hello, world!
```

您可以在以下存储库中找到完整示例：[https://github.com/eunomia-bpf/inline-hook-demo](https://github.com/eunomia-bpf/inline-hook-demo)

## inline hook 的实现

让我们来看一下`inline_hook`函数的实现。这是一个非常基本的实现，适用于x86_64、ARM64和ARM32。这不是一个完整的实现，但应该足够让您入门。

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
        // 在原始函数的开头写入一条跳转指令。
    *((unsigned char *)orig_func + 0) = 0xE9; // 跳转指令
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
    // 构造一个指向钩子函数的分支指令。
    // 在ARM中，分支指令的指令是0xEA000000 | ((<offset> / 4) & 0x00FFFFFF)
    // 由于在ARM中PC每次前进4个字节，因此需要将偏移量除以4
    int offset = ((intptr_t)hook_func - (intptr_t)orig_func - 8) / 4;
    int branch_instruction = 0xEA000000 | (offset & 0x00FFFFFF);

    // 将分支指令写入原始函数的开头。
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
    // 存储函数的原始字节。
    memcpy(orig_bytes, orig_func, SIZE_ORIG_BYTES);

    // 使内存页可写。
    mprotect(get_page_addr(orig_func), getpagesize(),
         PROT_READ | PROT_WRITE | PROT_EXEC);

    inline_hook_replace_inst(orig_func, hook_func);

    // 使内存页只可执行。
    mprotect(get_page_addr(orig_func), getpagesize(),
         PROT_READ | PROT_EXEC);
}

void remove_hook(void *orig_func)
{
    // 使内存页可写。
    mprotect(get_page_addr(orig_func), getpagesize(),
         PROT_READ | PROT_WRITE | PROT_EXEC);

    // 还原函数的原始字节。
    memcpy(orig_func, orig_bytes, SIZE_ORIG_BYTES);

    // 使内存页只可执行。
    mprotect(get_page_addr(orig_func), getpagesize(),
         PROT_READ | PROT_EXEC);
}
```

我们首先将目标函数的原始字节保存在`orig_bytes`数组中。然后，我们使用`mprotect`使内存页可写。接下来，我们将目标函数的前几条指令替换为跳转到钩子函数的跳转指令。最后，我们恢复内存页的权限到原始状态。`get_page_addr`计算对齐为页的地址。`inline_hook`通过存储原始字节和修改指令来设置钩子。`remove_hook`撤销更改。

钩子安装根据处理器架构的不同而有所差异。

在x86_64上，我们将目标函数的开头替换为跳转指令，以重定向到我们的钩子函数。

```c
#define SIZE_ORIG_BYTES 16
static void inline_hook_replace_inst(void *orig_func, void *hook_func) {
    // 在原始函数的开头写入一个跳转指令。
    *((unsigned char *)orig_func + 0) = 0xE9; // 跳转指令
    *((void **)((unsigned char *)orig_func + 1)) =
        (unsigned char *)hook_func - (unsigned char *)orig_func - 5;
}
```

请注意，在ARM32中，程序计数器(PC)通常比指令超前两个指令，因此在计算偏移量时，我们要减去8（2个指令 * 4字节/指令）。这可能在不同的ARM版本或模式（Thumb vs ARM等）之间有所不同，因此请根据目标的具体情况进行相应调整。

此外，您需要将 SIZE_ORIG_BYTES 从16增加到20，因为 ARM 中的最小分支指令是 4 个字节，您即将替换 5 个指令。这是因为分支指令使用相对偏移量，并且无法确定钩子函数与前一指令之间的距离。如果您的函数和钩子彼此之间的距离在 32MB 以内，您只需用一个分支指令替换前4个字节，无需更改其余部分。

```c
#define SIZE_ORIG_BYTES 20
static void inline_hook_replace_inst(void *orig_func, void *hook_func) {
    int offset = ((intptr_t)hook_func - (intptr_t)orig_func - 8) / 4;
    int branch_instruction = 0xEA000000 | (offset & 0x00FFFFFF);
    *(int *)orig_func = branch_instruction;
}
```

类似于 ARM32，ARM64 使用 ARM 指令集。然而，有一些 ARM64 的差异和特殊之处需要考虑。例如，分支指令的编码不同，并且由于较大的地址空间，您必须为不能通过单个分支指令到达的更大偏移量创建一个跳板。跳板应该靠近原始函数，以便通过分支指令到达，并从那里加载钩子函数的完整64位地址。

```c
#define SIZE_ORIG_BYTES 32
static void inline_hook_replace_inst(void *orig_func, void *hook_func) {
    int offset = ((intptr_t)hook_func - (intptr_t)orig_func) / 4;
    // 检查偏移是否在分支指令的允许范围内。
    if (offset < -0x2000000 || offset > 0x1ffffff) {
        printf("Offset %d out of range!\n", offset);
        exit(1);
    }
    // 构造并写入分支指令。
    uint32_t branch_instruction = 0x14000000 | (offset & 0x03ffffff);
    *((uint32_t*)orig_func) = branch_instruction;
}
```

您可以在以下存储库中找到完整示例： [https://github.com/eunomia-bpf/inline-hook-demo](https://github.com/eunomia-bpf/inline-hook-demo)

## 限制

了解 inline hook 的限制能更好地理解其中的挑战和影响，特别是在现代操作系统和编程环境的背景下。演示代码非常简单，不能在生产中使用，实际的 inline hook 由于要处理多种情况，因此要复杂得多。

### 1. 操作系统的安全机制

现代操作系统部署了各种安全机制，以防止对正在执行的代码进行恶意或意外的修改：

- 数据执行预防（DEP）：DEP旨在防止从进程的数据段（如栈或堆）运行代码。 inline hook 通常需要执行写入这些段的代码，这可能会被DEP阻止。
- 地址空间布局随机化（ASLR）：ASLR会对系统和应用文件使用的内存地址进行随机化。这使得 inline hook 的过程变得更加复杂，因为目标函数的准确地址可能会在每次应用程序或系统重新启动时发生变化。
- 代码签名和完整性检查：一些操作系统和应用实施了代码签名和完整性检查。这些机制可以检测到对代码的修改，包括 inline hook ，可能会阻止修改后的应用程序执行或将其标记为恶意。

### 2. 编译器优化

现代编译器采用了各种优化，可能会干扰 inline hook ：

- 函数内联：编译器可以内联函数，这意味着函数的代码会直接插入到每个调用它的地方，而不是保留为单独的函数。这可能会消除 inline hook 所依赖的一致的函数入口点。
- 指令重排序和优化：编译器可能重新排序指令或优化函数的结构，在与 inline hook 设置的假设不一致的方式下，可能导致崩溃或未定义的行为。

### 3. 多线程和并发执行

- 线程安全：在多线程应用程序中，确保正确应用钩子而不中断当前执行的线程可能是具有挑战性的。存在一种风险，即一个线程正在执行被钩住的函数，而另一个线程正在应用钩子。
- 重入问题：如果被钩住的函数或钩子本身是可重入的（可以同时从多个线程调用），会复杂化 inline hook 的过程。必须小心处理这样的情况，以避免死锁或不一致的程序状态。

### 4. 硬件和体系结构的特点

- 指令集差异：不同的处理器具有不同的指令集和执行模型。例如，ARM和x86处理器在处理指令的方式上存在显着差异，使得撰写通用的 inline hook 过程更加复杂。
- 指令长度变化：指令的长度可能会变化（特别是在像x86这样的可变长度指令集中），这使得确定安全地覆盖多少字节而不影响后续指令变得困难。

## 总结

了解 inline hook 可以在软件安全、测试和调试等领域提供实质性帮助。它提供了一种在运行时改变和控制程序行为的途径。虽然它很强大，但也有其局限性，需要小心处理。
总之，虽然 inline hook 是强大的工具，但应该谨慎使用，理解并且具备良好的系统架构知识。

希望您喜欢这次探索 inline hook 之旅。编程愉快！
