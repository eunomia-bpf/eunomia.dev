# 为什么 BPF arena 的一个页被释放后仍会保持映射，又是什么决定用户态拿到新页还是 SIGBUS？

**简短回答：** arena 不是一个只有一个所有者的小堆。里面的一个页有两个独立的产生者（BPF 侧的 `bpf_arena_alloc_pages` 和用户态缺页的 `arena_vm_fault`），而内核只用一张区间位图记住哪些槽位是空闲的。两个产生者都从这张位图里取槽位并装页；只有 `bpf_arena_free_pages` 会反向走一遍，而且它必须按顺序做完四件事——清掉 vmalloc 里的 PTE、刷掉内核 TLB、把该页从所有映射过它的用户 VMA 里 zap 掉，最后才把这个区间还给位图。所以用户态缺页装进来的页会一直驻留、并一直计入 `memlock`，直到某个 BPF 程序释放这段区间或整个 arena 被销毁。用户态在缺页时到底拿到新页还是一个信号，只由一个 map flag 决定：不带 `BPF_F_SEGV_ON_FAULT` 时缺页会装一个新分配的零页，带上它时缺页返回信号而不分配。

## 机制

### 1. 位图记录的是「空闲」槽位，两个产生者都去清它

arena 的 range tree 是一张「未分配」槽位的位图。map 创建时整段区间被置位，每次分配则清掉它要占的槽位：

```c
range_tree_init(&arena->rt);
err = range_tree_set(&arena->rt, 0, attr->max_entries);
```

极性从函数名就能看出来：`is_range_tree_set()` 在所请求的整段区间仍空闲时返回 0，而 `range_tree_clear()` 把一段区间标记为已被占用。`arena_adjust_tree()` 是唯一决定 BPF 侧分配要吃掉哪些槽位的地方——调用者给了偏移就用它，否则用 `range_tree_find()` 做一次最佳适配：

```c
if (uaddr) {
        ret = is_range_tree_set(&arena->rt, *pgoff, page_cnt);
        if (ret)
                return ret;
        return range_tree_clear(&arena->rt, *pgoff, page_cnt);
}
ret = range_tree_find(&arena->rt, page_cnt);
```

用户态缺页路径是第二个产生者，它在同一把自旋锁下一次取一个槽位，在装 PTE **之前**先清掉位图里的那一位。两个产生者因此不会把同一页发给两方：仲裁者是位图里的一位，而不是页上的锁。

还有第三种状态：预留。`bpf_arena_reserve_pages()` 只清位图、不装页，这是让一侧告诉另一侧「这里不要放页」的手段；这样预留出来的槽位会让之后落在该地址的 `bpf_arena_alloc_pages()` 失败，也让用户态缺页走错误路径而不去分配。

### 2. 释放是四步，而顺序就是全部要点

`arena_free_range()` 并不会在调用者开口时立刻把页放掉，它按下面的顺序走：

1. `apply_range_clear_cb()` 逐个槽位清掉内核 PTE 并收集 `struct page`，同时跳过 arena 自己的 scratch 页。
2. 对收集到的每个页做 `lock_page()`/`unlock_page()`，等掉任何已经找到该页、正在进行中的 `arena_vm_fault()`，之后才允许复用这个 PTE。
3. 用 `flush_tlb_kernel_range()` 刷掉内核 TLB。
4. 这时才用 `zap_pages()` 从每个映射过该区间的用户 VMA 里 zap 掉，然后 `__free_page()`。位图的那一位最后才恢复。

源码里的注释把这层契约说得很直白：「区间一直留在 range tree 里，直到它从 vmalloc 区域和所有用户 vma 里都被解除映射，这样在过期 TLB 项或用户 PTE 还在的时候没人能把它填上。」把其中任意两步对调，就会出现「位图里已空闲、但仍有映射可达」的窗口，也就是位图本来要防的重复分配。单页情况只是被优化过、流程并未不同：`arena_free_range()` 在引用计数显示该页被用户映射时对大页走批量 zap，只有一页时直接调用 `zap_pages()`。

### 3. 两个释放入口，一个推迟执行

`bpf_arena_free_pages()` 是 kfunc，所以可以被不能睡的上下文调用。释放路径很长——要拿自旋锁、走页表、刷 TLB，`zap_pages()` 里还要拿 `mmap_read_lock()`——于是不可睡的那一版选择推迟：分配一个小小的 `arena_free_span` 记录，挂到无锁链表上，再排 `irq_work`，由它调度一个 work 在进程上下文里执行 `arena_free_range()`。可睡的版本当场执行。

这个推迟是有代价的，源码也写明：如果从不可睡上下文里连那条小记录都分配不出来，这些页会「有意保持不可达（泄漏）直到 arena 被销毁」。也就是说，在原子上下文或持锁上下文里释放、又恰逢内存压力时，调用者可以把这次释放观察成整轮运行内的空操作。

还有一点要注意：`bpf_arena_free_pages()` 会只保留地址的低 32 位、按页对齐取整，再把区间裁剪到已登记的用户 VMA 之内。落在 VMA 之外的地址，或者起点在 VMA 之前的区间，是被静默截断而不是报错的。

### 4. 用户态看到什么，由一个 map flag 决定

`arena_vm_fault()` 有三种结局，走哪一种由 `BPF_F_SEGV_ON_FAULT` 和位图决定，而不是单看地址：

- 页已经在（BPF 分配的、或之前缺页装上的），加锁返回。
- 该槽位在位图里仍空闲、且没有设置那个 flag，就分配一个清零的新页，同时装进内核映射和用户映射，并把这次分配记到 map 的 memcg 与 `nr_pages` 上。这就是惰性路径：用户态填一页就是一次分配，在 `memlock` 里与 BPF 侧的分配无法区分。
- 如果该槽位在位图里是**预留**状态（对它 `is_range_tree_set()` 为真），缺页返回 `VM_FAULT_SIGBUS`——区间存在，但没有页。
- 设置了 `BPF_F_SEGV_ON_FAULT` 且没有页时，缺页返回 `VM_FAULT_SIGSEGV`，不分配任何东西。

缺页处理还会因为与地址无关的原因直接失败：拿自旋锁失败时它返回 `SIGBUS` 而不重试，因为这个处理函数从未持有 `mmap_lock`，重试会把它再拿一次。

### 5. BPF 侧自己的缺页是一张 scratch 页，而不是真正缺页

BPF 程序解引用一个从未分配过页的 arena 地址时，并不走通常意义的缺页。`bpf_arena_handle_page_fault()` 从栈上找出正在运行的程序，确认地址落在 4 GiB 带加上部 guard 之内，把 arena 的 scratch 页装进那个空 PTE，并把一次违例报给程序的 stderr：

```c
bpf_stream_printk(ss, "ERROR: Arena %s access at unmapped address 0x%lx\n",
                  write ? "WRITE" : "READ", addr);
```

这次访问会对着 scratch 页照常完成。它是一个诊断，不是正确性机制：两个不同的错误偏移会写同一张 scratch 页，读回来的值也不是程序的数据。scratch 页永远不会被区间释放放掉（两条清理路径都跳过它），也不计入 `memlock`。

因为 scratch 页占住了该槽位的 PTE，真正的分配器必须把它拆掉：`apply_range_set_cb()` 在一个循环里重试装入，而在没有原子 PTE 装入原语的架构上，它期望槽位是空的。源码里对此的说明是：缺少 `ptep_try_set()` 的架构同时也没有接上 `bpf_arena_handle_page_fault()`，所以它们从来不会有 scratch 页需要拆。

## 在真实内核上验证与排查

1. 用 `memlock` 看，而不是看 map 的 RSS。内核真正计入的是 `arena->nr_pages << PAGE_SHIFT`，由 `/proc/<pid>/fdinfo/<map_fd>` 暴露出来。刚创建的 arena 报 0——scratch 页被排除在外——selftest 断言的就是这个序列：0，17 页 BPF 分配后变成 `17 * PAGE_SIZE`，从起点释放一页后是 16 页，接着 6 页、0 页，用户态触了 8 页后变成 `8 * PAGE_SIZE`，再被 BPF 释放回 0。

2. 用一对程序区分三种缺页结局。触碰一个未分配的页并看到 `memlock` 涨一页：这是惰性路径。用 `bpf_arena_reserve_pages()` 预留同一段再触碰：进程收到 `SIGBUS`，`memlock` 不动。用 `BPF_F_SEGV_ON_FAULT` 创建 map 再触碰新页：`SIGSEGV`。三者中任何一个给出别的结果，就说明 map 的 flag 走到了另一条分支。

3. 确认一次释放真的完成了，而不是变成了一条推迟记录。从可睡的程序里（`syscall` 程序是最省事的宿主）调用 `bpf_arena_free_pages()` 会在 `zap_pages()` 跑完之后才返回；从不可睡的上下文里调用则只把 span 排进队列就返回，连记录都分配不出来时返回得更早。此时要看 `memlock`，并在 workqueue 跑过之后再确认一次。

4. 确认 32 位偏移的契约。BPF 程序访问 arena 用的是 `kern_vm_start + 低32位(user_pointer)`，释放路径在按页对齐后也只保留低 32 位。用户 VMA 跨过 4 GiB 边界时 mmap 阶段就会被拒，原因就在这里：`map_extra` 必须让整段区间待在一个 32 位窗口内。

5. 预期的是加载期被拒，而不是运行期出错。非 GPL 兼容的程序根本不能调用内核函数；校验器在解析这些调用时就会以 “cannot call kernel function from non-GPL compatible program” 拒绝，kfunc 一次也不会执行，也就不会碰到任何 arena 页。

## 限制

位图仲裁的是槽位，不是使用者。两个 BPF 程序可以同时持有同一个 arena 里的指针，而释放路径不检查调用者是否还拥有这段区间：释放一段另一个程序仍在用的区间会把它从那个程序脚下解除映射，下一次访问就是一次 scratch 页违例。所有权是共享这个 arena 的程序之间的约定，不是被强制的属性。

释放的页数越过 VMA 会被裁剪，起点落在 VMA 之外的释放则静默返回。用起来很宽容，排查时却帮不上忙：由一个过期偏移算出的错误指针不会产生任何错误，只会让释放比预期少，差值要到之后某次分配落在调用者没预料的位置时才暴露出来。

scratch 页还意味着一个错误的 arena 指针不一定会致命。写到未分配偏移上的程序不会破坏真实数据，但也拿不到访问点上的任何错误——只有一行 stderr——所以一个忽略自己日志的程序会带着没有被记录下来的违例继续跑。

最后，这套机制受 4 GiB 可寻址带限制，也受「用户 VMA 只在第一次 `mmap()` 时确定」这一事实限制。第二个进程必须用 `MAP_FIXED` 映射到同一地址，而大小不同的 VMA 会被 arena 拒绝，于是多进程共享是一个部署期的决定，而不是 map 之后还能协商的事情。

## References

- [BPF arena 实现，内核源码](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/arena.c) —— 区间位图、`arena_adjust_tree()`、`arena_free_range()`、`zap_pages()`、推迟释放，以及 scratch 页的违例路径。
- [range tree，内核源码](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/range_tree.c) —— 文件头注释说明这棵树记录的是 arena 的**未分配**槽位，以及 `range_tree_set` 与 `range_tree_clear` 的拆分/合并规则。
- [arena kfunc 声明，内核源码](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/helpers.c) —— `bpf_arena_alloc_pages`、`bpf_arena_free_pages` 与 `bpf_arena_reserve_pages`，以及向不可睡版本的转调。
- [arena map 类型与 flag，uapi 头文件](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/uapi/linux/bpf.h) —— `BPF_MAP_TYPE_ARENA`、`BPF_F_SEGV_ON_FAULT`、`BPF_F_MMAPABLE` 与 `map_extra` 的地址契约。
- [arena 内存计量 selftest](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/testing/selftests/bpf/prog_tests/arena_mem_usage.c) —— 跨 BPF 分配、单页与批量释放、用户态缺页、释放已缺页页的完整 `memlock` 序列。
- [arena 分配与释放测试程序](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/testing/selftests/bpf/progs/arena_mem_usage.c) —— 用 `syscall` 程序驱动 `bpf_arena_alloc_pages()` 和 `bpf_arena_free_pages()`，偏移与页数由调用者给出。
- [arena mmap 契约 selftest](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/testing/selftests/bpf/prog_tests/arena_mmap.c) —— 对超出 map 长度的 VMA、地址 0、跨 4 GiB 的区间以及回绕区间的拒绝。
- [arena kfunc 参数形式 selftest](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/testing/selftests/bpf/progs/arena_kfunc.c) —— 用 arena 全局变量占住第 0 页、单页分配与释放，以及 32 位标量与完整用户地址两种参数形式。

## 今日社区讨论

本次通过获批的只读归档查看了两个归档选择加入的 Slack 工作区，共三个白名单频道，日窗口内只有一条消息。公开的 bpf 邮件列表归档与 r/eBPF 作为可见浏览器页面查看了；今天实质性的来源是邮件列表，而 r/eBPF 最近的帖子是项目发布而不是提问。两个白名单 Discord 工作区共十个频道本次没有查看：没有开 Discord 会话，所以它们在这里的缺席是覆盖缺口，不是「没人说话」。归档里那一条消息是一个二进制模式查找 kfunc 的提案；它是真实的开放问题，但本页回答下面的 arena 所有权问题，因为那个查找话题已经有一篇专门回答，而 arena 讨论才是今天真正出现设计分歧的地方。

**一个共享页归谁，以及所有者停下来的那一刻会发生什么。** 今天最清楚的设计争论是关于在主机与其 guest 之间共享 arena 页。第一版把一块 pinned arena 暴露成一个定长的文件，并把 guest 那一份做成一个 NUMA 节点；评审找出的两个结构性问题正好是上面机制所预示的：节点就是普通的 guest 内存，所以 guest 的调度器退出后它的 arena 页回到 guest 的 buddy allocator，而主机还能写它们；另外块粒度意味着一页共享页要占 16 MB guest 内存，把设计上限压到每主机调度器几百个 guest。替换方案把所有权问题反转过来：arena 只增加一个原语，映射它自己没分配的页，每个 import 由**供应方**而不是 arena 用一棵区间树记录，一次调用整块释放，不计入 arena 的内存计量，并且供应方消失后 guest 再触碰就报 `SIGBUS`。两个设计值得成对读，因为第二个列出的那些性质，正是上面释放路径对 arena 自有页必须做对的那份清单。

**「分配」对同一个页意味着两件不同的事。** 围绕用户态一侧还出现了一个相关且仍未解决的混淆。对于 BPF 从未碰过的页，用户态的一次写入就是一次分配：缺页处理取一个位图槽位、分配一页并把它计入。对于被程序用 `BPF_F_SEGV_ON_FAULT` 声明过的 arena，同一次写入是错误信号。两种行为只差一个 map flag，而失效形态很容易被误读——来自**预留**区间的 `SIGBUS` 与来自缺页处理内部分配失败的 `SIGBUS` 看起来完全一样，后者是内存压力的症状而不是设计错误。前后各看一次 `memlock` 是分辨二者最便宜的办法：错误信号不改变这个数，惰性分配恰好让它涨一页。

**一次不对程序安全性作任何承诺的校验器改造。** 窗口里最大的单条线程是对校验器主分析的 49 个补丁重写，用状态里携带的等式替换掉原来的路径跟随核心与标量演化分析。对一个要决定是否采用它的读者来说，有意思的是它自己的后续系列怎么描述：后续七个修复针对的都是「不会让校验器接受不安全程序」的 bug——不带 `CONFIG_NET` 的构建失败、路径敏感的运行凭空造出一条失败路径从而拒掉合法程序、日志只显示最后一个基本块而不是通往错误的路径，以及 offloaded 程序因精度原因被拒。真正重要的是给出的数字：约 6000 个来自其他项目的程序上，每程序平均指令数从约 18k 降到 1.7k，平均校验时间从约 23 ms 降到 5 ms，峰值状态数均值从 562 降到 18。配套的七个修复在生产校验器负载的 385k 个程序上测得判定不变、处理指令数也不变，这才是让重写可以发布的那个数字，而不是加速比。

**循环仍是两侧意见不一致的边界。** 一组相关修复在一次提交里解决了三个独立的循环交互 bug：绑定会移动时要给被跳过的行程数标成「猜测」，32 位值回绕时也是如此，而仅仅因为帧不够而被丢弃的路径不该被算成失败。三个 bug 的形态相同——分析把没能证明的值当成已经证明了。这与把 scratch 页读回的结果当成数据属于同一类错误，也正是上面的 arena 页坚持「仲裁者是位图而不是页」的原因。