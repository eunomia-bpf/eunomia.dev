import type { Locale } from "../lib/site-data";

type DailyReportDisclosureProps = {
  locale: Locale;
  placement: "top" | "footer";
};

type DirectionItem = {
  title: string;
  body: string;
};

type DirectionCopy = {
  gapsTitle: string;
  gapsIntro: string;
  gaps: DirectionItem[];
  ideasTitle: string;
  ideasIntro: string;
  ideas: DirectionItem[];
};

function canonicalReportPath(path: string): string {
  return path.startsWith("/zh/") ? path.slice(3) : path;
}

export function DailyReportDisclosure({ locale, placement }: DailyReportDisclosureProps) {
  const copy =
    locale === "zh"
      ? {
          title: "AI 生成，未经人工审核",
          top:
            "本页由自动化研究与写作工具生成，尚无人逐项核验引用、事实、数字和建议。请把它当作研究起点，回看一手来源，并在用于生产系统或重要决策前独立验证。",
          footer:
            "披露：本页由 AI 生成，未经过人工编辑审核，仍可能包含遗漏、误读或错误。请通过页面中的一手来源核验关键结论，并可通过源码链接提交修正。"
        }
      : {
          title: "AI-generated and not human-reviewed",
          top:
            "This page was produced with automated research and writing tools. No human editor has verified every citation, fact, metric, or recommendation. Treat it as a starting point, check the primary sources, and independently validate it before using it in production or important decisions.",
          footer:
            "Disclosure: this page is AI-generated and has not received human editorial review. Omissions, misreadings, and errors may remain. Verify important claims against the primary sources and use the source link to propose corrections."
        };

  return (
    <aside
      role="note"
      className={[
        "border border-amber-300 bg-amber-50 px-5 py-4 text-amber-950",
        placement === "top" ? "mb-8" : "mt-12"
      ].join(" ")}
    >
      <p className="font-semibold">{copy.title}</p>
      <p className="mt-2 text-sm leading-6">{placement === "top" ? copy.top : copy.footer}</p>
    </aside>
  );
}

const traceDirections: Record<Locale, DirectionCopy> = {
  en: {
    gapsTitle: "Where current work is still weak",
    gapsIntro:
      "Existing tracing and agent-diagnosis systems establish useful mechanisms, but they do not yet answer the retention problem end to end.",
    gaps: [
      {
        title: "There is no ground-truth benchmark for evidence selection.",
        body:
          "Most tracing papers measure overhead, compression, anomaly recall, or diagnosis after a trace already exists. Agent-diagnosis benchmarks likewise tend to assume that the full trajectory is available. We still lack a corpus that starts from a high-fidelity execution, hides evidence under a fixed budget, and measures whether an investigator can reach the correct conclusion."
      },
      {
        title: "Evidence value has no shared metric.",
        body:
          "Bytes retained and events per second are easy to count, but they do not reveal whether one retained fact distinguishes two plausible root causes. Without a decision-level utility metric, retention policies can optimize storage while preserving the wrong information."
      },
      {
        title: "Semantic anchors are not portable across runtimes.",
        body:
          "Agent frameworks, MCP tools, shells, containers, and system-level monitors expose different identities and boundaries. There is no widely used schema that connects a model decision to delegated processes, object versions, authority, external effects, and final outcomes across those layers."
      },
      {
        title: "Adaptive capture can create hidden bias and privacy debt.",
        body:
          "A monitor that escalates collection after a model flags risk changes which data survives. That selection policy can overfit known incidents, expose sensitive content, and make fleet-wide statistics misleading unless the decision itself is versioned and auditable."
      }
    ],
    ideasTitle: "Promising directions with academic and production value",
    ideasIntro:
      "The most useful next work would turn evidence retention from a plausible architecture into a measurable systems problem.",
    ideas: [
      {
        title: "EvidenceBudgetBench: evaluate retention policies at equal cost.",
        body:
          "Build a consented corpus of real tool-using agent runs with complete system evidence, normal tasks, natural failures, injected faults, policy violations, and hidden incident classes. Replay head sampling, tail sampling, flight recorders, anomaly-only capture, and evidence portfolios under the same runtime, storage, privacy, and review budget. Measure population-estimation error, incident recall, critical-step localization, counterfactual discrimination, and analyst effort. This would provide an academically clean benchmark and a practical way for operators to choose a policy before deployment."
      },
      {
        title: "An active evidence controller based on value of information.",
        body:
          "Instead of using fixed triggers, let an online controller decide which next observation would most reduce uncertainty between competing explanations. The controller should reserve random exploration, obey privacy and cost constraints, and record why it expanded collection. A strong study would compare static policies with active capture on long-horizon incidents and test whether the controller improves diagnosis without turning rare behavior into permanent surveillance."
      },
      {
        title: "A portable evidence schema with system-level adapters.",
        body:
          "Define stable work-unit, authority, object-version, effect, and outcome records that can be emitted by agent runtimes and independently reconstructed through eBPF, process, file, network, Git, and MCP adapters. The research question is how much cross-runtime diagnosis and policy checking survives when semantic labels are incomplete or wrong. The production payoff is interoperability between agent frameworks and existing observability stacks."
      }
    ]
  },
  zh: {
    gapsTitle: "现有研究还缺什么",
    gapsIntro: "现有 tracing 与 Agent 诊断工作已经提供了不少有效机制，但还没有把“有限预算下该留下什么证据”作为端到端问题解决。",
    gaps: [
      {
        title: "缺少专门评测证据选择的真值数据集。",
        body:
          "多数 tracing 论文评测开销、压缩率、异常召回率，或者在完整 trace 已经存在的前提下评测诊断。Agent 诊断 benchmark 也通常默认完整轨迹可用。当前还缺少这样的语料：先保留一次执行的高保真真值，再按固定预算隐藏部分证据，最后测量调查者能否得到正确结论。"
      },
      {
        title: "证据价值没有统一指标。",
        body:
          "保留了多少字节、每秒记录多少事件都容易计算，但它们不能说明某条证据是否真的能区分两个合理的根因解释。如果没有面向决策的效用指标，系统可能把存储优化得很好，却保留了错误的信息。"
      },
      {
        title: "语义锚点无法跨 runtime 复用。",
        body:
          "Agent framework、MCP 工具、shell、容器和系统级监控各自暴露不同的身份与边界。目前没有一套广泛使用的 schema，能够把模型决策、委托进程、对象版本、权限、外部效果和最终结果跨层连接起来。"
      },
      {
        title: "自适应采集会带来隐蔽的偏差与隐私债务。",
        body:
          "当监控模型判断某段行为可疑后扩大采集范围，它也改变了哪些数据最终会被保存。这种选择策略可能过拟合已知事故、暴露敏感内容，并让全局统计产生偏差，因此采集决策本身也需要版本、预算和审计记录。"
      }
    ],
    ideasTitle: "兼具学术价值与生产价值的方向",
    ideasIntro: "下一步最有价值的工作，是把证据保留从一个合理架构变成可重复、可比较的系统问题。",
    ideas: [
      {
        title: "EvidenceBudgetBench：在相同成本下比较保留策略。",
        body:
          "构建一个经授权的高保真工具型 Agent 执行语料，覆盖正常任务、自然故障、注入故障、策略违规以及触发器未知的新型事故。然后在相同运行时、存储、隐私和审阅预算下离线重放 head sampling、tail sampling、flight recorder、仅异常保留和 evidence portfolio。评测总体估计误差、事故召回、关键步骤定位、反事实区分能力和调查成本。它既可以形成清晰的学术 benchmark，也能帮助生产团队在部署前选择策略。"
      },
      {
        title: "基于信息价值的主动证据控制器。",
        body:
          "与其只依赖固定 trigger，不如让在线控制器判断下一条什么观测最能缩小多个根因解释之间的不确定性。控制器需要保留随机探索预算，遵守隐私和成本限制，并记录为什么扩大采集。高质量研究应比较静态策略与主动采集在长链路事故上的诊断收益，同时验证它不会把所有少见行为都变成长期监控对象。"
      },
      {
        title: "带系统级适配器的可移植证据 schema。",
        body:
          "定义稳定的 work unit、权限、对象版本、效果和结果记录，让 Agent runtime 主动输出，同时让 eBPF、进程、文件、网络、Git 与 MCP 适配器独立重建。研究重点是：当 framework 的语义标签缺失或错误时，跨 runtime 诊断与策略检查还能保留多少能力。生产价值则是让不同 Agent framework 接入现有 observability 基础设施。"
      }
    ]
  }
};

const parallelDirections: Record<Locale, DirectionCopy> = {
  en: {
    gapsTitle: "Where current work is still weak",
    gapsIntro:
      "Parallel-agent systems already expose concurrency, but the research and runtime contracts around shared effects remain incomplete.",
    gaps: [
      {
        title: "Benchmarks undercount semantic conflicts.",
        body:
          "Coding-agent evaluations usually measure task completion, textual merge conflicts, or test success. They rarely include two patches that touch different files while violating one API invariant, two actions that consume one quota, or two valid local decisions that jointly fail the user's goal."
      },
      {
        title: "Tools do not expose machine-checkable effect contracts.",
        body:
          "A runtime often sees a shell command, browser action, or HTTP request rather than a declared read set, write set, budget consumption, reversibility class, and idempotency boundary. Without that information, conflict detection starts after effects have already escaped."
      },
      {
        title: "Global acceptance is weaker than local verification.",
        body:
          "Tests and guardrails commonly validate one worker or one tool call. Few systems express a global outcome such as keeping code and deployment configuration consistent, choosing exactly one supplier, or sending one externally visible message only after all prerequisites commit."
      },
      {
        title: "Concurrency control and authority are studied separately.",
        body:
          "Database techniques reason about versions and conflicts, while agent-security systems reason about principals and permissions. Parallel agents need both at the same commit point because a technically serializable effect may still rely on stale, consumed, or incorrectly delegated authority."
      }
    ],
    ideasTitle: "Promising directions with academic and production value",
    ideasIntro:
      "A useful research agenda should improve correctness without reducing every agent workflow to serial execution.",
    ideas: [
      {
        title: "An effect transaction layer for agent tools.",
        body:
          "Design adapters that translate file, Git, database, cloud, browser, and MCP operations into a common effect manifest: reads, writes, consumed capabilities, preconditions, reversibility, and external visibility. Workers prepare results in parallel; a coordinator validates versions, shared constraints, authority, and global acceptance before commit. Evaluate it against worktrees, full serialization, locks, and optimistic validation on coding, cloud, and business workflows. This has systems novelty because the transaction spans heterogeneous opaque tools, and immediate production value because it can sit between existing agents and their tools."
      },
      {
        title: "A semantic-conflict benchmark with hidden shared resources.",
        body:
          "Create tasks whose branches merge cleanly and whose local tests pass, while the combined result violates an API contract, quota, approval, ordering rule, or user-level objective. Publish a ground-truth conflict graph and measure final correctness, false serialization, repair cost, token waste, and duplicate external effects. Such a benchmark would expose the gap that today's merge and task-success metrics miss."
      },
      {
        title: "Adaptive concurrency control with deterministic commit checks.",
        body:
          "Use a learned risk model only to choose among full parallelism, optimistic validation, locking, and serialization based on effect type, conflict history, abort cost, and authority sensitivity. Keep the final commit checks deterministic wherever versions, schemas, policies, and idempotency can decide. The research question is whether this hybrid policy preserves most parallel speedup while preventing semantic failures at lower coordination cost than one fixed isolation level."
      }
    ]
  },
  zh: {
    gapsTitle: "现有研究还缺什么",
    gapsIntro: "并行 Agent 已经成为 runtime 能力，但共享效果的正确性契约仍然不完整。",
    gaps: [
      {
        title: "现有 benchmark 低估了语义冲突。",
        body:
          "Coding Agent 评测通常关注任务完成率、文本 merge conflict 或测试是否通过，很少覆盖这样的情况：两个 patch 修改不同文件却共同破坏一条 API 约束，两个动作消费同一笔 quota，或者两个局部合理的决定组合后违背用户目标。"
      },
      {
        title: "工具缺少机器可检查的效果契约。",
        body:
          "Runtime 往往只能看到 shell 命令、浏览器动作或 HTTP 请求，而看不到声明式的读取集合、写入集合、预算消耗、可逆性和幂等边界。缺少这些信息时，系统通常要等副作用已经发生后才发现冲突。"
      },
      {
        title: "全局验收明显弱于局部验证。",
        body:
          "测试与 guardrail 通常验证单个 worker 或单次 tool call，很少表达整项任务的全局结果，例如代码与部署配置必须一致、只能选择一个供应商，或者所有前置条件提交后才能发送一次对外消息。"
      },
      {
        title: "并发控制与权限控制仍被分开研究。",
        body:
          "数据库技术关注版本与冲突，Agent 安全系统关注主体与权限。并行 Agent 需要在同一个提交点同时处理两者，因为一项可以串行化的技术操作，仍可能依赖已经过期、被消费或错误委托的权限。"
      }
    ],
    ideasTitle: "兼具学术价值与生产价值的方向",
    ideasIntro: "真正有用的研究方向，应在提高正确性的同时，避免把所有 Agent 工作流都退化成串行执行。",
    ideas: [
      {
        title: "面向 Agent 工具的效果事务层。",
        body:
          "为文件、Git、数据库、云 API、浏览器和 MCP 操作设计适配器，把它们转换成统一的效果清单：读取、写入、被消费的 capability、前置条件、可逆性和外部可见性。Worker 并行准备，协调器在提交前检查版本、共享约束、权限和全局验收。评测应覆盖 coding、cloud 和业务工作流，并与 worktree、完全串行、锁和 optimistic validation 比较。它的学术价值来自跨异构 opaque tool 的事务语义，生产价值则来自可以直接部署在现有 Agent 与工具之间。"
      },
      {
        title: "包含隐藏共享资源的语义冲突 benchmark。",
        body:
          "构造一组分支可以 clean merge、局部测试也通过，但组合结果会违反 API 契约、quota、审批、顺序规则或用户目标的任务。公开真值 conflict graph，并评测最终正确率、误串行化、修复成本、token 浪费与重复外部效果。这类 benchmark 可以补上当前 merge 指标与任务成功率看不到的缺口。"
      },
      {
        title: "带确定性提交检查的自适应并发控制。",
        body:
          "学习模型只负责根据效果类型、历史冲突、abort 成本和权限敏感度，在完全并行、optimistic validation、加锁和串行之间选择；最终提交尽量由版本、schema、policy 与幂等规则做确定性验证。核心研究问题是，这种混合策略能否保留大部分并行加速，同时以低于固定隔离级别的协调成本阻止语义错误。"
      }
    ]
  }
};

function directionCopyForPath(path: string, locale: Locale): DirectionCopy | null {
  const canonicalPath = canonicalReportPath(path);
  if (canonicalPath === "/research/agent-trace-evidence-budget/") {
    return traceDirections[locale];
  }
  if (canonicalPath === "/research/parallel-agent-effect-serializability/") {
    return parallelDirections[locale];
  }
  return null;
}

export function DailyReportResearchDirections({ locale, path }: { locale: Locale; path: string }) {
  const copy = directionCopyForPath(path, locale);
  if (!copy) {
    return null;
  }

  return (
    <>
      <section className="mt-12" aria-labelledby="daily-report-gaps">
        <h2 id="daily-report-gaps">{copy.gapsTitle}</h2>
        <p>{copy.gapsIntro}</p>
        {copy.gaps.map((gap) => (
          <div key={gap.title}>
            <h3>{gap.title}</h3>
            <p>{gap.body}</p>
          </div>
        ))}
      </section>
      <section className="mt-12" aria-labelledby="daily-report-ideas">
        <h2 id="daily-report-ideas">{copy.ideasTitle}</h2>
        <p>{copy.ideasIntro}</p>
        {copy.ideas.map((idea) => (
          <div key={idea.title}>
            <h3>{idea.title}</h3>
            <p>{idea.body}</p>
          </div>
        ))}
      </section>
    </>
  );
}
