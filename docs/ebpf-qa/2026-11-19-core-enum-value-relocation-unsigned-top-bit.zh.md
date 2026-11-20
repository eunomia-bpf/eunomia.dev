# 为什么带符号位已置位的无符号枚举成员会让 CO-RE 枚举值重定位拒绝整个目标文件，修复又必须接受哪些形式？

**简短回答：** BTF 把每个 32 位枚举成员存在一个有符号字段里，枚举本身的符号性记录在类型 `info` 字段的第 31 位（kind flag）。CO-RE 在计算枚举值时把这个有符号字段直接赋给 64 位变量，于是*无符号*枚举里 `0x80000000` 及以上的成员会被符号扩展：`0xefffffff` 变成 `0xffffffffefffffff`。编译器为无符号枚举发出的是零扩展形式，重定位因此无法把指令里的立即数与它从本地类型算出的值对上，整个目标文件在进入校验器之前就被拒绝；目标侧的值也会被按错误的形式打补丁，只要程序把它当 64 位用就出错。又因为 clang 和 GCC 发出的形式不同，修复必须同时接受两种形式，而不是简单地换一个扩展方式。

## 机制

### 1. BTF 把值和它的符号性分开存放

`BTF_KIND_ENUM` 的每个成员是一个名字加一个 32 位有符号值，而枚举自身的符号性在类型 `info` 的第 31 位，文档里写作「0 表示无符号，1 表示有符号」。超出 32 位的值用另一个 kind：`BTF_KIND_ENUM64`。

```c
struct btf_enum {
        __u32   name_off;
        __s32   val;
};
```

所以无论枚举本身是不是无符号，这个字段都是有符号的，`val` 只是一个位模式：成员 `SKB_DROP_REASON_SUBSYS_MASK = 0xffff0000` 存成 `-65536`。想知道 C 语义上的值，工具必须依据 kind flag 决定扩展方式。libbpf 在另外两条路径上早就这么做了——数据 dumper 和枚举类型的 map 字段都走一个按 flag 扩展的辅助函数：

```c
static inline __u64 btf_enum_value(const struct btf_type *t, const struct btf_enum *e)
{
        return btf_kflag(t) ? (__s64)e->val : (__u32)e->val;
}
```

### 2. 重定位路径跳过了这一步

CO-RE 枚举值重定位的计算只在一个小函数里。取值分支长这样：

```c
t = btf_type_by_id(spec->btf, spec->spec[0].type_id);
if (btf_is_enum(t))
        *val = btf_enum(t)[spec->spec[0].idx].val;
else
        *val = btf_enum64_value(btf_enum64(t) + spec->spec[0].idx);
```

`*val` 是 `__u64`，源是 `__s32`，代码不看 kind flag，于是赋值对整个 32 位枚举做了符号扩展。`enum64` 那支取的是完整 64 位值，不受影响。整个缺陷就在这里：同一个库里的两个辅助函数，用两种方式扩展同一个字段。

### 3. 错值在哪里变成被拒绝的目标文件

算出来的值被当作本地一侧的*期望值*，再与编译器实际生成的指令比对。`ld_imm64` 形式是字面比较：

```c
imm = (__u32)insn[0].imm | ((__u64)insn[1].imm << 32);
if (res->validate && imm != orig_val) {
        pr_warn("prog '%s': relo #%d: unexpected insn #%d (LDIMM64) value: got %llu, exp %llu -> %llu\n", ...);
        return -EINVAL;
}
```

`got` 是指令里的值，`exp` 是 libbpf 算出来的值，第三个数是目标侧本来要打成的值。两者不一致时重定位返回 `-EINVAL`，libbpf 拒绝整个程序，一个程序都加载不了。这条路径完全不涉及校验器，所以错误看起来像构建期或加载期的不兼容，而不像内核版本问题——这正是它容易被误判的原因。

### 4. 内核跑的是同一份代码

`kernel/bpf/Makefile` 用 `tools/lib/bpf/relo_core.c` 编译出 `relo_core.o`，是同一个文件；内核里的重定位入口逐条处理程序的记录：

```c
err = bpf_core_apply(&ctx, &core_relo, i, &prog->insnsi[core_relo.insn_off / 8]);
```

当记录随加载请求一起送进内核而不是在用户态应用时走的就是这条路径——轻量级 skeleton 携带的生成式加载器就是如此：生成器把每条 `bpf_core_relo` 记录复制进加载属性。因此修复必须落在共享文件里，而它调用的辅助函数必须在两侧都存在：用户态那份头文件和内核那份。前面那个辅助函数是随 dumper 与 map 字段的改动一起加入的，已经在 BPF 树里，但内核侧头文件还没有，所以重定位系列也一并把它加到内核头文件。

### 5. 两个编译器发出的形式不同

clang 按枚举成员自己的 C 类型发值。下面三种情况都在本机编译并从指令流读回：

- 无符号枚举里的 `0x80000000u`：指令里是 `2147483648`，即零扩展的 `0x0000000080000000`。
- 无符号枚举里的 `0xffffffffu`：指令里是 `4294967295`，即零扩展的 `0x00000000ffffffff`。
- `-1`，其枚举在 BTF 里标为无符号：指令里是 `-1`，即符号扩展的 `0xffffffffffffffff`。

前两条是零扩展，与今天重定位算出的 `0xffffffff80000000`、`0xffffffffffffffff` 必然不等。第三条是问题的另一半：它是 kind flag 意义上的无符号枚举，clang 依然发出了符号扩展形式，因为成员自己的类型是 `int`。所以「按 kind flag 扩展」能修好前两条，却会弄坏第三条——而且这发生在 clang 编出的目标文件里，不只是 GCC 的问题。

GCC 没有这个选择余地：系列里写的是它的内建函数返回 `int`，因此每个 32 位值在指令里都是符号扩展。于是 GCC 编出的目标文件现在能加载，而只切换扩展方式的做法会把它拒掉。评审中的版本两种都接受：先按 kind flag 扩展；当本地值落在符号扩展形式上、且目标值能用 32 位表示时，把符号扩展形式作为第二种可接受的本地值，并按同一形式给目标打补丁。小于 `0x80000000` 的本地值在两种形式下相同，这时目标侧按目标枚举自己的 flag 扩展。

### 6. 内核里有多少枚举受影响

统计本机 7.3 vmlinux BTF：2909 个 32 位枚举类型、23 个 enum64 类型中，29 个无符号 32 位枚举带有 `0x80000000` 及以上的成员。其中包含 `init_ns_ino`（`IPC_NS_INIT_INO` 是 `0xefffffff`）、x86 的内存布局类型（`E820_TYPE_SOFT_RESERVED`，同样是 `0xefffffff`）、套接字丢包原因（`SKB_DROP_REASON_SUBSYS_MASK` 是 `0xffff0000`）、网络设备的私有 flag、缺页错误码的各个位，以及 ring buffer 的 busy 位。只要本地声明是同一枚举的无符号版本，对其中任一成员做值重定位就会触发；对同一枚举做字段偏移或类型重定位则不会。

## 验证与排查路径

1. 用一个枚举、一次调用复现：

   ```c
   enum uns_max { M1 = 0xffffffffu };
   __u64 v = bpf_core_enum_value(enum uns_max, M1);
   ```

   带调试信息编译后读指令：`llvm-objdump -d` 显示 `r0 = 4294967295 ll`，即零扩展的 `0x00000000ffffffff`。

2. 算出重定位期望的值。从目标文件自己的 BTF 里取同一个成员，按有符号 32 位扩展：`0xffffffffffffffff`。两者只在高 32 位不同。

3. 按这个形状读加载器的报错：对 BTF 里标为 `encoding=UNSIGNED` 的枚举，`got` 很小且是零扩展、`exp` 高 32 位全一，就是本缺陷，而不是目标不匹配。`bpftool` 对同一目标文件的 BTF dump 会同时显示编码和工具打印出的值。

4. 与「目标里没有该成员」区分开。目标里找不到同名成员时，重定位根本不拿值做校验：指令被投毒成一条调用非法函数号的指令，失败表现为 `invalid func unknown#195896080`（若投毒调用可达），或者完全不报错（若在死代码里）。存在性检查（`bpf_core_enum_value_exists`）不读值，两种情况都不受影响。

5. 工具链尚未修复时的绕法：在 32 位范围内比较或掩码——`(u32)bpf_core_enum_value(...)` 或 `& 0xffffffffu`——让符号扩展的补丁无害；如果目标文件必须在不带补丁的 libbpf 上加载，就别把高位无符号成员放进值重定位。

## 限制

修复只针对 32 位枚举；enum64 路径本来就取完整 64 位值。它修不了*语义*层面的错误：本地把枚举报成有符号、目标是无符号，算出的值仍然不同，而给目标打补丁时的扩展方式由目标枚举的 flag 决定。对小于 `0x80000000` 的值，「接受两种形式」的逻辑无法区分 clang 和 GCC，因为指令完全一样——所以单凭立即数推断编译器是不可靠的。另外本机没有 BPF 能力，这里的复现止步于指令字节、本地 BTF 编码和重定位算术，没有做完整加载。

## References

- [BTF 的枚举编码，内核文档](https://docs.kernel.org/bpf/btf.html) —— kind flag 表示符号性、有符号的 32 位成员字段，以及 enum64 作为独立 kind。
- [BTF 定义，uapi 头文件](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/uapi/linux/btf.h) —— 成员结构与 kind flag 位。
- [CO-RE 重定位核心，libbpf](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/lib/bpf/relo_core.c) —— 枚举值计算、立即数比较和投毒路径。
- [内核编译的同一份重定位文件](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/Makefile) —— 从 libbpf 源码生成的目标。
- [内核侧重定位入口](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/check_btf.c) —— 逐条应用程序的重定位。
- [CO-RE 辅助宏，libbpf](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/lib/bpf/bpf_core_read.h) —— 枚举内建函数的值形式与存在性形式。
- [重定位类型，内核文档](https://docs.kernel.org/bpf/llvm_reloc.html) —— 枚举值类型的 kind 与访问串。
- [生成式加载器的重定位记录，libbpf](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/lib/bpf/gen_loader.c) —— 记录如何随加载请求进入内核。
- [枚举扩展辅助函数，BPF 树](https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git/plain/tools/lib/bpf/btf.h) —— dumper 与 map 字段使用的按 flag 扩展。
- [map 字段的同一处归一化，libbpf](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/lib/bpf/btf_dump.c) —— 已经使用 flag 的 dumper 比较。
- [加入扩展辅助函数的提交](https://github.com/torvalds/linux/commit/89e7a42e3030) —— dumper 与 map 字段的改动，以及此前为什么会被符号扩展。
- [系列第三版，封面信](https://lore.kernel.org/bpf/20261119232647.3392045-1-yonghong.song@linux.dev/) —— 两种可接受的本地形式与目标打补丁规则。
- [系列第三版，libbpf 改动](https://lore.kernel.org/bpf/20261119232652.3392590-1-yonghong.song@linux.dev/) —— 确切报错输出、clang 与 GCC 的指令形式、受影响枚举的计数，以及内核头文件的补充。
- [系列的上一版](https://lore.kernel.org/bpf/20261119051443.1902163-1-yonghong.song@linux.dev/) —— 测试合入前的四补丁形态。
- [命名空间 inode 号，内核源码](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/dcache.h) —— 受影响枚举之一，含 `BIT(31)` 成员。
- [x86 内存布局类型，内核源码](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/include/asm/e820/types.h) —— `0xefffffff` 的保留软区间值。
- [GCC 的 BPF 内建函数](https://gcc.gnu.org/onlinedocs/gcc/BPF-Built-in-Functions.html) —— 系列称为返回 `int` 的枚举取值内建函数的文档。

## 今日社区讨论

通过已批准的只读归档读取了两个选择加入的 Slack 工作区，共三个白名单频道：24 小时窗口内没有消息，七天窗口里有三条——一条会议纪要通知、一条贡献者自我介绍，以及一个本站已经回答过的二进制模式查找问题。因此上面的问题来自 bpf 公共邮件列表归档，以普通公开页面方式读取；读取时该列表有 61 个线程根，索引里有 78 行的日期是 2026-11-19（UTC）。两个白名单 Discord 工作区（共十个频道）本轮没有覆盖：客户端返回的是未登录的登录页，因此记为不可访问，而不是安静。公开的从业者论坛读过了，窗口内没有新的技术问题，最新几篇是更早日期的两个项目公告和一条会议征稿。

### 同一个比特位，三个不同的所有者

上面的问题只是这个窗口里贯穿始终的一个形态：一个最高位被置位的 32 位值，被一段对宽度做了某种假设的代码处理。枚举这一例把错误放在重定位里，介于有符号的存储字段与编译器发出的无符号立即数之间。另外两个线程把错误分别放在 map 和校验器上。

最长前缀匹配 map 在 key 字节之后紧接着放 value，没有填充，所以 IPv4 的 value 从节点内第 4 个字节开始，只有 4 字节对齐。校验器检查 map value 访问的对齐时，只看 value 内部的偏移，并默认 value 本身至少 8 字节对齐——哈希 map 靠把 key 长度向上取整到 8 来保证这一点，而这个 map 类型没有。于是一个在 value 偏移 0 处执行的 64 位原子加能通过校验，在 arm64 上却会在 JIT 代码里触发对齐异常并 panic（缺少 LSE2 时任何 4 字节对齐地址都会，有 LSE2 时跨 16 字节 granule 也会）。提出的修复很朴素：给柔性数组加对齐，在分配、查找和内存统计里把 value 偏移向上取整到 8 字节，和哈希 map 保持一致。排查经验可以推广：任何把 value 放在变长 key 之后的 map，都要检查 value 偏移是否取整，不要以为校验器的对齐证明覆盖了基地址。

第三个是程序而不是值。一个把中断点的指令指针改掉、强制目标函数失败的辅助函数，已经在某一种探针形式上被拒绝，却在多附加点的用户态探针上仍然被接受。在那里附加后，任务会带着一个内核地址作为指令指针返回用户态：故障出现在任务的信号帧里，而产生它的寄存器状态属于内核。修复在链路创建时对入口和返回两种探针都拒绝这类程序，与单探针路径一致。

### 事件之间复用的 sample 数据，和没人写过的地址

一个三补丁系列修了软件事件在栈上构造的一份 sample 数据结构。第一个补丁让每个匹配的事件各自初始化：采样事件此前会继承上一个事件准备好的 sample id、时间戳、pid/tid 和调用链，有时还带着比随后写入更小的记录长度，从而覆盖环形缓冲里后面的记录。报告的症状很具体——两个计数为 1 的并采样会话，较老的那个失败，而且非特权用户的缺页事件能这样破坏 root 的全 CPU 会话，自 6.3 起就存在。第二个补丁阻止 BPF 程序读到未设置的地址：perf 事件数据上下文里的 `addr` 字段在几乎所有上下文切换和 CPU 时钟样本里返回残留栈内容（多为内核指针），自 6.1 起，只在「不置零」的栈初始化配置下可见。第三个补丁干脆去掉那 256 字节置零，把结构标记为未初始化，理由是每个字段都在被读之前写过；在单 CPU 上测得软件计数器给一次约 627 ns 的上下文切换增加 54 ns，改动后降到 44 ns。作者自己指出的残余边界是真实的：特权程序通过保留类型的转换仍能读到未准备的字段。

同一份结构还引出另一个不相关的线程。把逐 CPU 的寄存器快照换成栈上副本之后，自动评审指出：嵌套的程序会因此撞上输出辅助函数里的 WARN（配置了 panic_on_warn 的内核会直接 panic），而且提交说明里「不再返回 busy」的说法是错的。维护者同意这两点，要求在系列里一并去掉这个 WARN；其中一位还指出同一区域的另一个洞：带 build id 的栈采集可能通过完成 tracepoint 上的程序重入同一把锁，采到的标识符会消失。所以未解决的不是改动本身，而是它周围的递归约定。

### 网络补丁明确写出的两个边界

一个在发送侧增加 socket 赋值的系列把不做什么写得很清楚：赋值之后的软件分段不支持、已解密的发送报文不支持、回环会丢弃赋值。它对被赋值的 socket 取引用并安装一个负责释放的析构函数，这是把生命周期问题变成可识别的报文属性的常用做法。若打算用它做透明代理，这三个排除项在讨论性能之前就决定了设计是否成立。

一个接收侧的系列把「附加 XDP 即排除硬件接收合包」这条规则从各驱动搬到网络核心，因为驱动之间不一致：有的拒绝附加，有的在自己的 fixup 里清掉该特性，至少有一个在 XDP 挂载时仍保留合包。动机是让合包特性留在 wanted 集合里、并在卸载后恢复；对排查带 XDP 的接收负载的人来说，实用做法是在附加和卸载前后分别读 wanted 与 active 两组特性位，而不是相信接口自报的状态。

### 较小的条目

一个符号查找优化系列到了第四版，把内核符号表的线性扫描换成按名字有序的查找，并新增一个内核参数来改变索引标记的密度。工具侧，一个修复让内存 cgroup 测试的页缓存留在 overlayfs 之外；另一个库修复让 dumper 里针对有符号 32 位枚举的比较对无符号数据也成立，而它正是上面那个问题所依赖的同一个辅助函数。