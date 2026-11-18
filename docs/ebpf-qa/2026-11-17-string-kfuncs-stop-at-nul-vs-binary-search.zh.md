# 为什么内核里的 BPF 字符串查找遇到第一个零字节就停下，而二进制模式查找还需要什么？

**简短回答：** BPF 的字符串 kfunc 全都是 C 字符串语义。6.17 加入的 `bpf_strnstr` 和它周围这一族函数，都用 nofault 访问器逐字节读取被查找串和模式串，一旦任意一侧读到 `'\0'` 就立即返回，所以带内嵌零字节的缓冲区只会被搜到那个零字节为止。两侧还各自被限制在 65536 字节，超出的输入返回 `-E2BIG` 而不是给出部分结果。内核的 `lib/string.c` 里没有 `memmem`，也没有包住 `memchr` 加 `memcmp` 的 kfunc；`bpf_strncmp` helper 不能代替，因为它直接转给 `strncmp()`，而后者同样在 NUL 处跳出循环。今天要做二进制模式查找，只能自己写逐字节循环，而校验器的迭代上限又让「64 KB 缓冲区对 64 KB 模式」所需的双层循环无法通过。

## 机制

### 1. 校验器并不给字符串参数定类型

这些 kfunc 的每个参数都带 `__ign` 后缀，例如 `const char *s1__ign`。校验器用 `is_kfunc_arg_ignore()` 匹配这个后缀，再由 `get_kfunc_arg_type()` 把它变成 `ARG_IGNORE`。`check_func_arg()` 对它只做一件事：

```c
if (arg_type == ARG_IGNORE)
        return 0;
```

寄存器被当作源操作数读一次，除此之外什么都不检查：不看指针类型、不看大小、不看对齐、也不需要任何显式选择加入。安全性全部来自实现本身——它先用 `copy_from_kernel_nofault_allowed()` 校验一次地址，然后在 `guard(pagefault)` 之下用 `__get_kernel_nofault()` 逐字节读取。在 x86-64 上，这个前置检查会拒绝低于 `TASK_SIZE_MAX + PAGE_SIZE` 的地址、vsyscall 页和非规范地址，这也正是自测里用户态指针在 x86-64 上期望 `-ERANGE`、在地址空间重叠因而无法前置拒绝用户地址的架构上期望 `-EFAULT` 的原因。实际后果是：字符串字面量、只读 map 值、以及 ring buffer 样本里的指针，三者对它完全等价——kfunc 分不清它们，也不知道其中任何一个有多长。

这也是这些 kfunc 被两道门限制的原因。它们注册在公共 kfunc 集合上，`btf_kfunc_is_allowed()` 会先查这个集合、再查按程序类型注册的集合，所以 XDP、TC、tracing 和 syscall 程序都能用。但它们每一个都标了 `KF_PERFMON`：

```c
BTF_ID_FLAGS(func, bpf_strnstr, KF_PERFMON);
```

装载器没有指针泄漏权限时，校验器会拒绝这次调用：

```c
if (is_kfunc_perfmon(&meta) && !env->allow_ptr_leaks)
```

实际含义是需要 `CAP_PERFMON`（或 `CAP_SYS_ADMIN`），因为这个 kfunc 会穿过一个校验器从未检查过的指针去读内存。非特权的生成式程序根本用不上。

### 2. NUL 终止符是算法的一部分，不是读取逻辑的意外

`__bpf_strnstr` 的主体是一次双方都按字节推进的扫描，其中有两处因零而退出。模式串一侧，在内层循环里：

```c
__get_kernel_nofault(&c2, s2 + j, char, err_out);
if (c2 == '\0')
        return i;
```

被查找串一侧，在读到模式字节之后再判断：

```c
__get_kernel_nofault(&c1, s1 + j, char, err_out);
...
if (c1 == '\0')
        return -ENOENT;
```

两处退出都是语义判断，不是边界检查。模式串里的零字节表示「模式到此结束」，于是当前位置算作命中。被查找串里的零字节表示「串到此结束」，于是整段剩余扫描被放弃，调用返回 `-ENOENT`——哪怕模式就藏在下一个字节。对 `{"\x01", "\x00", "A", "B"}` 用 `len` 为 4 去搜 `"AB"`，扫描会在下标 1 停下并回答不匹配，而这两个字节明明在偏移 2 处。`lib/string.c` 里的 `strnstr` 形状相同，原因不同：它对模式串调 `strlen()`、对候选窗口调 `memcmp()`，因此含零字节的模式串在比较开始前就被截断了。

### 3. 上限借用的是扩展属性长度限制，且两侧都用

外层循环上界和内层模式串上界都是 `XATTR_SIZE_MAX`，即 `include/uapi/linux/limits.h` 里的 65536：

```c
for (i = 0; i < XATTR_SIZE_MAX; i++) {
        for (j = 0; i + j <= len && j < XATTR_SIZE_MAX; j++) {
```

`len` 参数限制的是「匹配可以从被查找串的哪个位置开始」，而不是读多少：源码注释明确说明，s2 允许读到超出 `len` 一个字节，以覆盖「模式串是被搜索前缀的后缀」这种情况。当外层循环或模式串循环触到上限时，函数返回 `-E2BIG`，所以超过 64 KB 的被查找串得到的是一个错误，而不是「前 64 KB 里没有」。这个上限是对「一个 kfunc 可以在关闭缺页处理的内核路径里转多久」的刻意限制；它与任何具体缓冲区的大小无关，持有 2 MB ring buffer 样本的调用方也没有办法让它继续往后搜。

### 4. strncmp helper 同样停在零，而且要求 map 已冻结

`bpf_strncmp` 看上去像是二进制答案，其实不是。它的第三个参数是常量字符串，由 `check_arg_const_str()` 检查：寄存器必须指向某个对程序只读的 map 里的值，该值必须已冻结，偏移必须是校验器已知的常量，并且字符串必须在值内部以 NUL 结尾（`strnchr(..., 0)` 必须成功）。它的函数体只有一行：

```c
BPF_CALL_3(bpf_strncmp, const char *, s1, u32, s1_sz, const char *, s2)
{
        return strncmp(s1, s2, s1_sz);
}
```

而 `strncmp()` 在零处退出循环：

```c
if (c1 != c2)
        return c1 < c2 ? -1 : 1;
if (!c1)
        break;
```

于是两个二进制缓冲区只要在共享的零字节之前一致、之后不同，就会比较相等。UUID、哈希摘要、或中间带零字节的报文头，正是它会误判的情形，而且误判表现为「假命中」，不是报错。

### 5. 二进制查找需要什么，为什么循环上限决定了形态

最直接的用户态绕法是用 `bpf_for(i, start, end)`（即 `bpf_iter_num` 循环）遍历缓冲区并逐字节比较。比较本身很便宜，循环次数不是。`bpf_iter_num_new` 被校验器内联，内联序列会拒绝超过 `BPF_MAX_LOOPS` 的范围，而后者是 `include/linux/bpf.h` 里的 `8 * 1024 * 1024`：

```
r0 = (u32)end - (u32)start; if (r0 > BPF_MAX_LOOPS) goto e2big;
```

单趟扫过 64 KB 完全没问题。但模式查找本身是两趟嵌套——每个候选偏移都要比较一遍模式串——64 KB 被查找串配 64 KB 模式串就是 4 G 次字节比较，超过上限五百多倍，靠手工展开也塞不进去。这就是查找原语必须放在内核里、而不能写成宏的结构性原因：内层扫描必须是一次 kfunc 调用，而不能是一个 BPF 循环。

kfunc 一侧还得替调用方做完 dynptr 路径本来要做的事。`bpf_dynptr_slice()` 的长度必须是校验器常量，因为校验器把返回指针标成 `PTR_TO_MEM`，其 `mem_size` 取自该常量，并用 `parent_id` 绑到 dynptr 上；程序一旦写自己的栈缓冲区，这个切片就失效。对 skb 或 XDP 的 dynptr，只有请求范围线性时切片才是直接指针，否则在调用方没有提供缓冲区的情况下返回 `NULL`。所以一个收 dynptr 的查找 kfunc 必须自己调 `bpf_dynptr_check_off_len()`、把非线性区间拷进临时缓冲区、再在自己的字节上扫描——而这恰恰是今天的字符串 kfunc 做不到的，因为它们收的是不带长度的裸指针。

## 如何验证

在有这个 kfunc 的内核上，「是否存在」是 BTF 问题，「带内嵌零的缓冲区能不能搜到」是一个程序的问题：

```bash
bpftool btf dump file /sys/kernel/btf/vmlinux format raw | grep -c '^bpf_strnstr$'
```

一个 `syscall` 程序，全局 `char buf[] = "\x01\x00AB";`，函数体 `return bpf_strnstr(buf, "AB", 4);` 返回 `-ENOENT`；同一个程序换成 `"AABB"` 返回 1。这一族的现成测试已经编进 `test_progs`，所以 `./test_progs -t string_kfuncs` 会覆盖命中、未命中、空模式串和用户态指针几类用例；它们期望值里 `-ERANGE` 与 `-EFAULT` 的分裂，是确认地址检查走了哪条架构路径最省事的办法。

至于「不存在」这个论断，在任何一棵树上三次 grep 就能定案：`kernel/bpf/helpers.c` 里没有 `bpf_memmem` 也没有 `bpf_memcmp`，`lib/string.c` 和 `include/linux/string.h` 里都没有 `memmem`，而 `memchr` 与 `memcmp` 都在且都已导出。树里还能看出为什么没有 libc 兜底：LLVM 的 BPF 后端只放行 `memcpy`、`memset`、`memmove`、128 位整数辅助函数和 trap 这几种内建函数调用，其余内建函数除非打开实验性的 libcall 开关，否则直接编译失败，所以 `__builtin_memcmp` 必须被内联掉才能生成代码。

## 局限

「不存在」是快照，不是定律。一个 `bpf_memcmp` 加 `bpf_strpbrk` 的补丁集已经以 v2 上过邮件列表，而 7.4-rc3 的 `kernel/bpf/helpers.c` 里两个都不存在，所以承诺它们存在的页面很快就会过期；依赖某个名字之前请读树，不要读这页。同样的谨慎适用于这一族函数的版本史：`bpf_strnstr` 本身在 6.17 有、在 6.16 没有，忽略大小写的几个兄弟函数更晚，所以「这些 kfunc 会做 X」这类句子只相对于某个版本成立。

64 KB 的上限也不会因为改名而消失。任何查找 kfunc 都得为「在关闭缺页处理的内核路径里可以跑多久」选一个界，选 65536 就意味着大缓冲区至多只能被搜一部分，剩下的用 `-E2BIG` 如实回答。需要搜 1 MB 样本的调用方仍然要自己切窗口，而跨越窗口边界的匹配仍然是调用方的问题。

最后，这些都不会扩大非特权程序的能力。`ARG_IGNORE` 看起来松散，但 `KF_PERFMON` 这道门才是它安全的原因：这个 kfunc 能读任意内核内存，只有特权装载器才被允许提出这个请求。

## References

- [bpf_strnstr 及其所在的字符串 kfunc 族，内核源码](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/kernel/bpf/helpers.c) — 两处因 NUL 退出、以 `XATTR_SIZE_MAX` 为界的循环、`bpf_strncmp`，以及 `KF_PERFMON` 注册。
- [内核 strnstr 与 memchr，lib/string.c](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/lib/string.c) — kfunc 所模仿的 C 字符串扫描，以及 `memmem` 的缺席。
- [XATTR_SIZE_MAX，uapi limits](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/uapi/linux/limits.h) — 两侧上限所用的 65536。
- [BPF_MAX_LOOPS，include/linux/bpf.h](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/linux/bpf.h) — 内联后的 `bpf_iter_num_new` 所强制的 8 Mi 迭代上限。
- [kfunc 参数注解规则](https://docs.kernel.org/bpf/kfuncs.html) — `__ign`、`__sz`、`__k` 后缀，`KF_PERFMON` 标志，以及 kfunc 集合按程序类型注册。
- [bpf_strncmp helper 文档](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/uapi/linux/bpf.h) — 常量字符串第三参数及其写明的长度语义。
- [字符串 kfunc 自测](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/testing/selftests/bpf/progs/string_kfuncs_success.c) — 每次调用的期望结果，含空模式串与后缀用例。
- [邮件列表上的二进制比较空缺](https://lore.kernel.org/bpf/20260624093638.117062-1-lange_tang@163.com/) — v2 `bpf_memcmp` 与 `bpf_strpbrk` 的 cover letter：`bpf_strncmp` 会在空字节处停住，以及自测里的绕法。
- [带本地比较循环的 dynptr 拷贝自测](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/testing/selftests/bpf/progs/dynptr_success.c) — 内核自己的测试退回到的手写字节比较。

## 今天的社区讨论

本次通过批准的只读归档审查了两个已选择接入归档的 Slack 工作区，共三个在册频道。24 小时窗口里没有消息；七天窗口里有三条，其中一条就是本文回答的问题，另两条是没有技术问题的自我介绍，因此本页的问题取自七天窗口，而不是取自今天。公开的 bpf 邮件列表归档作为普通网页阅读，覆盖到 7.4-rc3 的树状态与 v2 补丁集的讨论。两个在册 Discord 工作区共十个频道，以及 r/eBPF，本次没有审查：它们都是可见浏览器界面，而本次没有打开浏览器会话，所以这里对它们应当读作覆盖缺口，而不是「很安静」。

**在非字符串缓冲区上做二进制查找。** 本文回答的缺口，也正是邮件列表从另一头在争的同一件事。提议的 `bpf_memcmp` kfunc 给出了三类具体负载——L2 过滤与转发用的 6 字节 MAC 比较、16 字节 UUID 与摘要白名单/黑名单、以及被校验器弄得很别扭的手写循环——并点明了本文记录的那个缺陷：`bpf_strncmp` 会在空字节处停下，因此会破坏二进制比较。配套的 `bpf_strpbrk` 针对同一问题的字符串侧，即挂载选项与 `/proc` 条目里的分隔符查找，那里 `strspn` 加 `strcspn` 要两次调用才做到一次调用本该做的事。仍然未定的不是「要不要」，而是「放哪儿」：这个原语是像字符串族其余成员一样收裸指针，还是收 dynptr、把 skb 与 XDP 分片处理放进 kfunc 里而不是丢给每个调用方；以及当缓冲区来自报文、而不是来自冻结的 map 值时，长度参数究竟意味着什么。

**「限长查找」里的长度到底指什么。** `bpf_strnstr` 的长度语义经过了好几个版本才定下来，原因正是本文验证部分依赖的那条边界。以 `len` 为界的查找必须接受「模式串恰好结束于 `len`」的匹配，这意味着要从模式串读出搜索前缀之外的一个字节并在做决定之前比较它；源码里为此保留了一行注释。由此产生的行为在两侧都很微妙：起点在 `len` 以内、但延伸超出 `len` 的匹配不会被报告；而比上限更长的模式串会由内层循环返回 `-E2BIG` 而不是 `-ENOENT`。写二进制变体的人继承同一个问题，而且不同于字符串情形，他不能拿「被查找串到底了」当作停止的理由——这正是整件事的要点。

**失败语义随架构不同。** 自测里编码了一个本页必须谨慎表述的分裂：把 NULL 或用户空间指针交给字符串 kfunc，在地址空间不重叠的架构（含 x86-64）上返回 `-ERANGE`，在其他架构上返回 `-EFAULT`。在 s390 上，底层 nofault 访问器对 NULL 不触发异常而直接返回，所以整组测试在那里被编掉。给调用方的实用教训不是把错误码映射成原因——`-ERANGE`、`-EFAULT` 与 `-ENOENT` 都来自同一次调用——而是把三者都当成「扫描没有完成」，并先在调用方一侧校验指针与长度。

**特权边界究竟在哪里。** 因为字符串参数被校验器忽略，能力检查就是程序与任意内核读取之间唯一的屏障，讨论也倾向于保持现状：这些 kfunc 带 `KF_PERFMON`，而非 GPL 程序根本不能调用内核函数。对一个新的查找 kfunc 来说，同一个问题只是下沉了一层：基于 dynptr 的变体严格比基于裸指针的更安全，但它能被调用的程序类型也更少，而需求最强的恰恰是报文路径。