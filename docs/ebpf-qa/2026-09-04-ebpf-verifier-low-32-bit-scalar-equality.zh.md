# 为什么只在低 32 位成立的关系会被 eBPF 验证器丢失？

**简短回答：** eBPF 寄存器宽度是 64 位，而一条 32 位指令只会建立一种关于低半部分的特定关系。执行 `w7 = w6` 时，目标值会零扩展，所以 `r7` 等于 `zero_extend(low32(r6))`，不一定在完整 64 位上等于 `r6`。32 位到 64 位的符号扩展又是另一种关系：目标高半部分由第 31 位重复填充。如果验证器只能表达全宽相等，继续保留旧等价关系会不可靠，因此只能丢弃它。之后即使分支已经约束了一个寄存器的低 32 位，这项知识也无法传播到另一个寄存器。

这是一种精度限制，不代表程序已经被证明不安全。只要抽象状态无法证明所需边界或返回值约束，验证器就必须拒绝。当前有一个 RFC 提议分别记录零扩展与符号扩展的低 32 位等价关系，但方案仍在审查中。在目标内核具备等价能力之前，应让分支直接约束后续实际使用的转换结果。

## 运行时事实比“64 位相等”更窄

每个通用 eBPF 寄存器都是 64 位，但 ISA 区分 `ALU64` 与 32 位 `ALU` 操作。[eBPF 指令集规范](https://docs.kernel.org/bpf/standardization/instruction-set.html)把 32 位寄存器 move 定义为：

```text
r7 = (u32)r6
```

这里的转换不能省略：低 32 位被复制，`r7` 的高 32 位被清零。如果 `r6` 是 `0xffff_ffff_0000_0001`，那么 `r7` 会变成 `0x0000_0000_0000_0001`。两者的低半部分相等，但作为 64 位整数并不相等。

`MOVSX` 又有不同契约。执行 32 位到 64 位符号扩展时，目标高半部分由低半部分的第 31 位重建：该位为一则高半部分全为一，否则全为零。因此，下面三条声明彼此不同：

- `rA == rB` 在完整 64 位上成立；
- `low32(rA) == low32(rB)`，且 `high32(rB) == 0`；
- `low32(rA) == low32(rB)`，且 `rB == sign_extend(low32(rA))`。

把三者混为一谈，验证器可能错误接受不安全程序；如果验证器只能保存第一种关系，就必须在 subregister 转换后保守地忘记关系。

## 为什么丢失关系会导致误拒绝

看一段简化后的指令：

```text
r6 = unknown_64_bit_scalar
w7 = w6
if w6 != 0 goto reject
if w7 == 0 goto ok
```

第一个分支的 fall-through 路径已经证明 `r6` 的低 32 位为零。因为 `w7 = w6` 复制了这些位并零扩展，所以 `r7` 必然是 64 位常量零；第二个分支在运行时可直接判定。

验证器不会执行具体值，而是符号化地记录每个寄存器的有符号、无符号边界，以及表示已知位与未知位的 `tnum`。[验证器文档](https://docs.kernel.org/bpf/verifier.html#register-value-tracking)说明了分支如何收窄这些状态。验证器还会记录相关值之间的 identity，使一个副本上学到的知识有时能传播到另一个副本。

如果 identity 只表示“两个 64 位标量相等”，它就不能跨过 `w7 = w6`：两者的高半部分本来就可能不同。第一个分支收窄 `w6` 后，验证器已经没有理由更新 `r7`，于是可能认为 callback 返回值超出允许区间，把 errno-or-zero 结果重新放宽为很大的有符号范围，或者在循环中探索过多状态。这些是后续症状，低半部分关系丢失才是机制。

编译器输出会决定是否触发问题。看起来等价的源代码可能生成不同组合的 `MOV32`、`MOVSX` 和 64 位操作。当前内核讨论报告称，包含更多符号扩展的输出更容易暴露缺口，而另一种编译输出可能直接优化掉转换。因此，真正有用的证据是实际 BPF 指令流，而不只是 C 源码。

## 一个可靠的低 32 位 link 必须记录什么

正在讨论的[低 32 位标量 link RFC](https://lore.kernel.org/bpf/20260814231945.3884596-1-vineet.gupta%40linux.dev/T/#t)提议保留共享 scalar identity，并附带明确的关系类型。概念上，零扩展 link 表示：

```text
dst.low32  = src.low32
dst.high32 = 0
```

符号扩展 link 则表示：

```text
dst.low32  = src.low32
dst.high32 = repeat(src.bit31, 32)
```

当分支收窄共享低半部分时，验证器可以按相应规则重建目标的 64 位范围与已知位。它不能把源寄存器未知的高半部分复制给零扩展目标，也不能把符号扩展目标误当成零扩展。

这种 link 还必须参与所有“两个验证器状态是否兼容”的判断。验证器发现此前已接受的状态能够安全覆盖当前状态时，会剪枝停止继续探索。[内核验证器的剪枝文档](https://docs.kernel.org/bpf/verifier.html#pruning)说明，比较范围包括寄存器与 spill 到栈上的状态。如果剪枝时忽略 link 是全宽、零扩展还是符号扩展关系，就可能合并会产生不同实际值的状态；这会从精度问题升级为 soundness bug。

审查时至少要检查四个边界：

1. **创建：** 只有指令语义确实保证关系时才能创建 link。
2. **传播：** 一个低半部分被收窄后，只更新逻辑上必然成立的事实，不能凭空推导源寄存器高半部分。
3. **失效：** write、spill、fill、算术或 cast 不再保留关系时，必须清除或替换 link。
4. **状态比较：** 在路径剪枝前，exact-state 与 safe-subset 比较必须纳入 link 类型和 identity。

当前内核源码用 [`struct bpf_reg_state`](https://github.com/torvalds/linux/blob/master/include/linux/bpf_verifier.h)保存抽象寄存器状态，并在 [`kernel/bpf/states.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/states.c)中实现状态比较与剪枝。判断具体内核树支持什么，应查看这些源码，而不是根据版本号猜测。

## 怎样诊断这类拒绝

应从被拒绝的 object 与实际内核开始，因为源代码里出现 cast 并不能证明编译后仍是哪一种指令。

1. **保存完整 verifier log。** 找到 range 首次意外放宽，或关联寄存器停止同步收窄的位置。最终报错通常远晚于精度丢失点。
2. **反汇编 BPF object。** 确认相关 copy 是 `MOV32`、32 位 `MOVSX`、64 位 `MOV` 还是其他 ALU 操作；同时确认后续 comparison 使用 32 位还是 64 位 jump。
3. **分别追踪高低两半。** 每一步都写明 `low32`、目标高半部分的构造规则，以及分支真正约束的寄存器。不能把 `w7 = w6` 简写成 `r7 = r6`。
4. **只把编译器差异当成线索。** 如果一种优化输出可以加载、另一种不行，应 diff 实际 BPF 指令。它可以定位转换模式，却不能证明被拒绝的程序本身不安全。
5. **检查运行内核实现。** 在其 verifier state 与 selftest 中查找低 32 位关系支持。RFC 已发布或编译器较新，都不能证明功能已经进入该内核。
6. **最小化时保留转换。** 最小程序必须保留 move、收窄分支与失败 use。把 subregister 操作优化掉的 reduction 已不是同一个验证器问题。

有效证据包括 copy 与 branch 的准确指令位置、两者前后的寄存器状态、编译器与优化级别，以及运行内核 commit；不需要采集原始生产数据。

## 源码层面的实用规避方式

目标内核尚不能保留这种关系时，应让证明不依赖跨越有损 subregister link 的传播。

- **直接比较转换后的值。** 如果后续使用零扩展或符号扩展目标，也应直接对这个目标执行 range check。
- **只规范化一次并保留一个值。** 用符合真实语义的 `u32` 或 `s32` 变量保存结果，再让 branch 与最终 return、index 或 loop condition 使用同一变量。
- **避免无意义的符号往返。** 如果数值域确实是无符号，就保持无符号；若负值有业务意义，不能仅为迎合验证器而改变 signedness。
- **显式约束 callback return。** 在最终决策点把宽中间值转换到 callback 文档要求的返回范围，不要期待早期 alias 上的约束自动传播。
- **每次改写后检查生成指令。** 优化器可能合并变量或重新引入 extension，源码修改本身不是问题已经消失的证据。

不要为了消除拒绝而随意加入 mask 或 truncation，除非它们保留程序原本的数值域。改变负错误码、环绕行为或循环终止条件的“验证器规避”本身就是功能 bug。

## 怎样验证内核侧修复

完整修复不能只靠一个正例。selftest 至少应覆盖：

- 零扩展与符号扩展 copy；
- 从 source 与 destination 触发的收窄，同时不推导无关高位；
- 32 位与 64 位宽度下的相等、不等分支；
- spill/fill 与 overwrite 后的失效；
- link 与 constant delta 的组合；
- 循环收敛与状态剪枝；
- 同一 checkpoint 上不兼容的 link 类型；
- 多种编译器代码生成模式下应接受与应拒绝的程序。

还必须运行完整 BPF selftest，而不只是新增正例。提高精度可能让更多程序通过，也可能改变剪枝、循环探索或 stale-link 行为。每个新增通过都应能由预期的低 32 位关系解释，所有负例也必须继续因相同安全理由被拒绝。

## 参考资料

- [BPF 邮件列表 RFC：追踪低 32 位标量等价关系](https://lore.kernel.org/bpf/20260814231945.3884596-1-vineet.gupta%40linux.dev/T/#t)
- [Linux 内核文档：eBPF 指令集语义](https://docs.kernel.org/bpf/standardization/instruction-set.html)
- [Linux 内核文档：验证器寄存器值追踪](https://docs.kernel.org/bpf/verifier.html#register-value-tracking)
- [Linux 内核文档：验证器状态剪枝](https://docs.kernel.org/bpf/verifier.html#pruning)
- [Linux 内核 BPF 设计问答：32 位 subregister 要求](https://docs.kernel.org/bpf/bpf_design_QA.html#q-bpf-32-bit-subregister-requirements)
- [Linux 内核源码：`struct bpf_reg_state`](https://github.com/torvalds/linux/blob/master/include/linux/bpf_verifier.h)
- [Linux 内核源码：验证器状态比较与剪枝](https://github.com/torvalds/linux/blob/master/kernel/bpf/states.c)
- [GitHub 文档：为 Copilot 添加仓库级自定义指令](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/add-custom-instructions/add-repository-instructions)

## 当日社区讨论

本次通过普通可见浏览器检查了全部 6 个批准社区和 15 个白名单频道或公开页面，所有目标均可访问。选中的验证器讨论在严格的过去 24 小时窗口内有新进展，因此没有使用七天回退。以下综合分析已删除参与者、账号、雇主、项目与频道身份、消息链接、精确时间、私有拓扑、原始日志和可回搜原文；没有保存原始 transcript，也没有执行回复、表情互动、私信、关注、邀请或管理操作。

### Subregister 精度是最强的技术问题

内核讨论重新聚焦一个验证器缺口：32 位 copy 保留低半部分，却改变高半部分的构造规则。实践症状是某个程序在运行时能够满足狭窄的 callback 返回值或循环边界，验证器却在零扩展或符号扩展后失去证明。审查重点不只是“让更多程序通过”，还包括 link flag 是否参与状态比较、stale relation 是否会被清除，以及 verifier log 能否显示新状态。这种关注很关键：错误剪枝可能导致误接受，而丢失关系只会误拒绝有效程序。

同一窗口内的其他活跃内核主题包括标量边界推导、map 操作整数溢出、borrowed reference 生命周期、JIT 内存检查、socket map 计数和带随机输入的 flaky selftest。它们的共同点是边界证据：算术宽度、引用所有权或非确定输入，都必须在每个验证器或测试状态转换处保持显式。

### 维护者讨论 AI 生成 PR 的审核护栏

一个可观测性项目讨论了怎样分类异常庞大、由机器生成的改动，同时不阻止真正有价值的贡献。暴露出的失败模式是 reviewer overload：很长的说明与巨大 diff 容易制造“材料很完整”的印象，却让 scope、已复现行为和测试证据更难辨认。讨论中的建议包括仓库级 review instruction，以及根据变更规模或结构给出 warning-only 的 metadata check。

这些信号可以帮助分流，却不能证明 patch 正确，也不能证明内容由 AI 生成。因此，可靠 gate 应要求有界的问题陈述、可复现行为、可独立审查的 commit，以及与声明改动直接对应的测试。基于规模的自动化宜用于分类或请求复核，不应直接拒绝。公开仓库指令可以统一预期，但最终判断仍需人工检查代码与证据。

### 其余目标可访问但较安静

批准范围内的项目帮助与功能区域没有新的窗口内技术问题。一个调度器支持区存在较早的安装规避方法，但已超出严格窗口，也无需作为回退。插桩讨论只有围绕稳定化工作的协作安排。eBPF 聊天频道最新的技术问题已经在更早的每日 Q&A 回答，论坛最新主题也重复了昨天的网络问题。这些是完成检查后的安静结果，不是把覆盖缺口报告成零活动。
