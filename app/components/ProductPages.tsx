import Image from "next/image";
import type { ReactNode } from "react";

import type { ReactPageLink } from "../lib/content/types";
import type { MkdocsHomeProject } from "../lib/content/mkdocs-config";
import type { Locale } from "../lib/site-data";
import { ContactCard, CredibilityStrip, StarBar, type StarRepo } from "./Credibility";

const ORG_STARS: StarRepo[] = [
  { repo: "bpftime", label: "bpftime" },
  { repo: "bpf-developer-tutorial", label: "Tutorials" },
  { repo: "eunomia-bpf", label: "eunomia-bpf" }
];

type ProductPageProps = {
  locale: Locale;
  links: ReactPageLink[];
  projects: MkdocsHomeProject[];
};

function linkMap(links: ReactPageLink[]): Map<string, ReactPageLink> {
  return new Map(links.map((link) => [link.key, link]));
}

function projectImage(projects: MkdocsHomeProject[], key: string): string | undefined {
  return projects.find((project) => project.key === key)?.image;
}

function SectionHeading({
  eyebrow,
  title,
  description
}: {
  eyebrow?: string;
  title: string;
  description?: string;
}) {
  return (
    <div className="max-w-3xl">
      {eyebrow ? (
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">{eyebrow}</p>
      ) : null}
      <h2 className="mt-3 text-2xl font-semibold tracking-normal text-ink md:text-3xl">{title}</h2>
      {description ? <p className="mt-3 text-base leading-7 text-slate-600">{description}</p> : null}
    </div>
  );
}

function LinkButton({ link }: { link: ReactPageLink }) {
  const primary = link.variant === "primary";
  const external = /^https?:\/\//.test(link.href);

  return (
    <a
      href={link.href}
      target={external ? "_blank" : undefined}
      rel={external ? "noopener" : undefined}
      className={[
        "inline-flex min-h-11 items-center rounded-md px-4 py-2 text-sm font-semibold transition",
        primary
          ? "bg-slate-950 text-white hover:bg-slate-800"
          : "border border-slate-300 bg-white text-slate-700 hover:border-slate-400 hover:text-ink"
      ].join(" ")}
    >
      {link.label}
    </a>
  );
}

function ActionRow({ links }: { links: Array<ReactPageLink | undefined> }) {
  const configuredLinks = links.filter((link): link is ReactPageLink => Boolean(link));

  return (
    <div className="mt-7 flex flex-wrap gap-3">
      {configuredLinks.map((link) => (
        <LinkButton key={`${link.label}:${link.href}`} link={link} />
      ))}
    </div>
  );
}

function CapabilityGrid({
  items
}: {
  items: Array<{
    label: string;
    title: string;
    description: string;
  }>;
}) {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {items.map((item) => (
        <article key={item.title} className="border border-slate-200 bg-white p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">{item.label}</p>
          <h3 className="mt-3 text-lg font-semibold tracking-normal text-ink">{item.title}</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">{item.description}</p>
        </article>
      ))}
    </div>
  );
}

function ProductEntry({
  eyebrow,
  title,
  description,
  href,
  image,
  imageAlt,
  links,
  visualLabel = "runtime plane",
  visualLines = ["observe.process()", "enforce.syscalls()", "protect.checkpoints()"]
}: {
  eyebrow: string;
  title: string;
  description: string;
  href?: ReactPageLink;
  image?: string;
  imageAlt?: string;
  links: Array<ReactPageLink | undefined>;
  visualLabel?: string;
  visualLines?: string[];
}) {
  const titleContent = (
    <h3 className="mt-2 text-2xl font-semibold tracking-normal text-ink">
      {href ? (
        <a href={href.href} className="transition hover:text-cyan-800">
          {title}
        </a>
      ) : (
        title
      )}
    </h3>
  );

  return (
    <article className="grid gap-6 border-t border-slate-200 py-7 first:border-t-0 md:grid-cols-[minmax(0,1fr)_16rem] md:items-center">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">{eyebrow}</p>
        {titleContent}
        <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">{description}</p>
        <ActionRow links={links} />
      </div>
      {image ? (
        <a href={href?.href} className="relative min-h-36 overflow-hidden rounded-md border border-slate-200 bg-slate-50">
          <Image
            src={image}
            alt={imageAlt ?? ""}
            fill
            sizes="16rem"
            className="object-contain p-4"
            unoptimized
          />
        </a>
      ) : (
        <div className="border border-slate-200 bg-slate-950 p-4 text-sm text-slate-100">
          <p className="font-mono text-xs uppercase tracking-[0.16em] text-cyan-200">{visualLabel}</p>
          <div className="mt-5 space-y-2 font-mono text-xs leading-6">
            {visualLines.map((line) => (
              <p key={line}>{line}</p>
            ))}
          </div>
        </div>
      )}
    </article>
  );
}

function Pipeline({
  stages
}: {
  stages: Array<{
    title: string;
    description: string;
  }>;
}) {
  return (
    <div className="grid gap-3 md:grid-cols-4">
      {stages.map((stage, index) => (
        <div key={stage.title} className="border border-slate-200 bg-white p-4">
          <p className="text-xs font-semibold text-cyan-700">{String(index + 1).padStart(2, "0")}</p>
          <h3 className="mt-3 text-base font-semibold tracking-normal text-ink">{stage.title}</h3>
          <p className="mt-2 text-sm leading-6 text-slate-600">{stage.description}</p>
        </div>
      ))}
    </div>
  );
}

function VisualPanel({
  children,
  image,
  imageAlt
}: {
  children?: ReactNode;
  image?: string;
  imageAlt?: string;
}) {
  return (
    <div className="relative min-h-64 overflow-hidden border border-slate-200 bg-slate-50">
      {image ? (
        <Image
          src={image}
          alt={imageAlt ?? ""}
          fill
          sizes="(min-width: 1024px) 32rem, 100vw"
          className="object-contain p-6"
          unoptimized
        />
      ) : (
        <div className="p-6">{children}</div>
      )}
    </div>
  );
}

export function ProductsLandingPage({ locale, links, projects }: ProductPageProps) {
  const linkByKey = linkMap(links);
  const bpftimeImage = projectImage(projects, "bpftime");
  const copy =
    locale === "zh"
      ? {
          eyebrow: "Products",
          title: "面向 runtime extension 和 Agent infra 的生产基础设施",
          description:
            "为需要 enterprise support、POC 和 production integration 的团队提供清晰采用路径。",
          mapEyebrow: "Product map",
          mapTitle: "选择适合的工程路径",
          mapDescription:
            "按生产目标组织：runtime extension、agent infrastructure 和 enterprise support。",
          bpftime:
            "Userspace eBPF runtime，面向低开销 tracing、GPU paths、custom runtime extension 和生产集成。",
          agent:
            "把 AgentSight 和 ActPlane 作为组合方向，覆盖 agent 行为观测、OS/eBPF 级约束和 checkpoint/restore 安全能力。",
          services:
            "固定范围咨询、POC、production hardening、performance tuning，以及 eBPF 或 agent infra 的定制集成。",
          buyersTitle: "适合的团队",
          buyersDescription:
            "适合需要把开源系统工程落到生产环境的 platform、security 和 AI infra 团队。",
          flowLabels: ["bpftime", "Agent infra", "企业支持"],
          buyers: [
            {
              label: "Platform / SRE",
              title: "低开销观测和 runtime extension",
              description: "把 uprobe、syscall、USDT、XDP 和 GPU 路径接入已有 tracing、profiling 或 runtime 平台。"
            },
            {
              label: "Security",
              title: "系统边界上的 agent 控制",
              description: "在进程、文件、网络、exec 和 checkpoint/restore 边界建立可审计、可执行的策略点。"
            },
            {
              label: "AI infra / AgentOps",
              title: "agent 行为证据和生产集成",
              description: "把 agent 行为从应用日志提升到 OS/runtime 级证据，支持 pilot、审计和后续平台集成。"
            }
          ]
        }
      : {
          eyebrow: "Products",
          title: "Production infrastructure for runtime extension and AI agent systems",
          description:
            "Support, pilots, and production integration for teams adopting Eunomia's open-source infrastructure.",
          mapEyebrow: "Product map",
          mapTitle: "Clear engineering paths",
          mapDescription:
            "Choose the path that matches the production problem: runtime extension, agent infrastructure, or engineering support.",
          bpftime:
            "A userspace eBPF runtime for low-overhead tracing, GPU paths, custom runtime extension, and production integration.",
          agent:
            "Agent infrastructure around AgentSight and ActPlane, spanning observability, OS-level enforcement, and checkpoint/restore safety.",
          services:
            "Fixed-scope consulting, POCs, production hardening, performance tuning, and custom eBPF or agent infrastructure integration.",
          buyersTitle: "Who it helps",
          buyersDescription:
            "Built for platform, security, and AI infrastructure teams that need open-source systems engineering to land in production.",
          flowLabels: ["bpftime", "Agent infra", "Enterprise support"],
          buyers: [
            {
              label: "Platform / SRE",
              title: "Low-overhead observability and runtime extension",
              description: "Connect uprobe, syscall, USDT, XDP, and GPU paths to existing tracing, profiling, or runtime platforms."
            },
            {
              label: "Security",
              title: "Agent control at system boundaries",
              description: "Create auditable and enforceable policy points across process, file, network, exec, and checkpoint/restore boundaries."
            },
            {
              label: "AI infra / AgentOps",
              title: "Operational evidence for agent behavior",
              description: "Move agent evidence beyond application logs into OS/runtime signals for pilots, audits, and platform integration."
            }
          ]
        };

  return (
    <section className="pb-16">
      <div className="grid gap-10 border-b border-slate-200 pb-12 lg:grid-cols-[minmax(0,1fr)_24rem] lg:items-center">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">{copy.eyebrow}</p>
          <h1 className="mt-4 max-w-4xl text-4xl font-semibold tracking-normal text-ink md:text-5xl">
            {copy.title}
          </h1>
          <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-600">{copy.description}</p>
          <CredibilityStrip locale={locale} className="mt-6" />
          <div className="mt-5">
            <StarBar repos={ORG_STARS} locale={locale} />
          </div>
          <ActionRow links={[linkByKey.get("bpftime"), linkByKey.get("contact")]} />
        </div>
        <VisualPanel>
          <div className="space-y-4">
            {copy.flowLabels.map((label, index) => (
              <div key={label} className="flex items-center gap-3 border border-slate-200 bg-white p-3">
                <span className="flex h-8 w-8 items-center justify-center rounded-md bg-slate-950 text-xs font-semibold text-white">
                  {index + 1}
                </span>
                <div>
                  <p className="text-sm font-semibold text-ink">{label}</p>
                  <p className="text-xs text-slate-500">
                    {locale === "zh" ? "开源基础设施和生产部署路径" : "open source plus production deployment"}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </VisualPanel>
      </div>

      <div className="py-12">
        <SectionHeading eyebrow={copy.mapEyebrow} title={copy.mapTitle} description={copy.mapDescription} />
        <div className="mt-6">
          <ProductEntry
            eyebrow="Runtime"
            title="bpftime"
            description={copy.bpftime}
            href={linkByKey.get("bpftime")}
            image={bpftimeImage}
            imageAlt="bpftime"
            links={[linkByKey.get("bpftime"), linkByKey.get("bpftime-github")]}
          />
          <ProductEntry
            eyebrow="Agent infrastructure"
            title="Agent Runtime Infrastructure"
            description={copy.agent}
            href={linkByKey.get("agent-infra")}
            links={[linkByKey.get("agent-infra")]}
            visualLabel="agent infra"
            visualLines={["observe.agent()", "enforce.policy()", "audit.runtime()"]}
          />
          <ProductEntry
            eyebrow="Services"
            title="Services / Enterprise Support"
            description={copy.services}
            href={linkByKey.get("services")}
            links={[linkByKey.get("services"), linkByKey.get("contact")]}
            visualLabel="delivery loop"
            visualLines={["review.architecture()", "prototype.integration()", "harden.production()"]}
          />
        </div>
      </div>

      <div className="border-t border-slate-200 py-12">
        <SectionHeading title={copy.buyersTitle} description={copy.buyersDescription} />
        <div className="mt-6">
          <CapabilityGrid items={copy.buyers} />
        </div>
      </div>

      <ContactCard locale={locale} contact={linkByKey.get("contact")} />
    </section>
  );
}

export function BpftimeProductPage({ locale, links, projects }: ProductPageProps) {
  const linkByKey = linkMap(links);
  const bpftimeImage = projectImage(projects, "bpftime");
  const copy =
    locale === "zh"
      ? {
          eyebrow: "Product / Runtime",
          title: "bpftime",
          description:
            "高性能 userspace eBPF runtime 和 extension framework，面向 production extension、observability 和 GPU-aware instrumentation。",
          whereTitle: "商业支持范围",
          whereDescription:
            "围绕 bpftime 开源 runtime 提供生产集成、性能调优和定制 runtime 工程支持。",
          useCasesTitle: "Use cases",
          supportTitle: "Commercial support",
          useCases: [
            "Low-overhead tracing",
            "Custom runtime extension",
            "uprobe / syscall / USDT / XDP / GPU paths",
            "Production integration"
          ],
          support: [
            "Enterprise support",
            "Integration with existing observability or runtime platforms",
            "Performance tuning and benchmarking",
            "Custom runtime / runtime extension engineering"
          ],
          capabilities: [
            {
              label: "Enterprise support",
              title: "生产集成和维护",
              description: "帮助团队把 bpftime 集成到已有 tracing、networking、sandbox 或 runtime extension 工作流中。"
            },
            {
              label: "Performance",
              title: "低开销观测和调优",
              description: "围绕 uprobe、syscall、USDT、XDP 或 GPU 相关路径做 benchmark、profiling 和性能优化。"
            },
            {
              label: "Custom runtime",
              title: "定制事件源和扩展",
              description: "为特定系统构建 attach path、helper、map、policy 或部署模型，而不强迫用户改动业务代码。"
            }
          ]
        }
      : {
          eyebrow: "Product / Runtime",
          title: "bpftime",
          description:
            "A high-performance userspace eBPF runtime and extension framework for production extension, observability, and GPU-aware instrumentation.",
          whereTitle: "Commercial support scope",
          whereDescription:
            "Support covers production integration, performance work, and custom runtime engineering around the open-source runtime.",
          useCasesTitle: "Use cases",
          supportTitle: "Commercial support",
          useCases: [
            "Low-overhead tracing",
            "Custom runtime extension",
            "uprobe / syscall / USDT / XDP / GPU paths",
            "Production integration"
          ],
          support: [
            "Enterprise support",
            "Integration with existing observability or runtime platforms",
            "Performance tuning and benchmarking",
            "Custom runtime / runtime extension engineering"
          ],
          capabilities: [
            {
              label: "Enterprise support",
              title: "Production integration",
              description: "Integrate bpftime into existing tracing, networking, sandboxing, or runtime extension workflows."
            },
            {
              label: "Performance",
              title: "Low-overhead tuning",
              description: "Benchmark, profile, and tune uprobe, syscall, USDT, XDP, and GPU-related execution paths."
            },
            {
              label: "Custom runtime",
              title: "New event sources",
              description: "Build attach paths, helpers, maps, policies, and deployment models for specific production systems."
            }
          ]
        };

  return (
    <section className="pb-16">
      <div className="grid gap-10 border-b border-slate-200 pb-12 lg:grid-cols-[minmax(0,1fr)_28rem] lg:items-center">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">{copy.eyebrow}</p>
          <h1 className="mt-4 text-5xl font-semibold tracking-normal text-ink md:text-6xl">{copy.title}</h1>
          <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-600">{copy.description}</p>
          <CredibilityStrip locale={locale} osdi={linkByKey.get("osdi")} className="mt-6" />
          <div className="mt-5">
            <StarBar repos={[{ repo: "bpftime", label: "bpftime" }]} locale={locale} />
          </div>
          <ActionRow
            links={[
              linkByKey.get("bpftime-docs"),
              linkByKey.get("bpftime-github"),
              linkByKey.get("osdi"),
              linkByKey.get("support")
            ]}
          />
        </div>
        <VisualPanel image={bpftimeImage} imageAlt="bpftime runtime architecture">
          <p className="font-mono text-xs uppercase tracking-[0.16em] text-cyan-700">bpftime runtime</p>
        </VisualPanel>
      </div>

      <div className="grid gap-6 py-12 lg:grid-cols-2">
        <article className="border border-slate-200 bg-white p-6">
          <h2 className="text-xl font-semibold tracking-normal text-ink">{copy.useCasesTitle}</h2>
          <ul className="mt-5 space-y-3 text-sm leading-6 text-slate-600">
            {copy.useCases.map((item) => (
              <li key={item} className="flex gap-3">
                <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-700" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </article>
        <article className="border border-slate-200 bg-white p-6">
          <h2 className="text-xl font-semibold tracking-normal text-ink">{copy.supportTitle}</h2>
          <ul className="mt-5 space-y-3 text-sm leading-6 text-slate-600">
            {copy.support.map((item) => (
              <li key={item} className="flex gap-3">
                <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-700" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </article>
      </div>

      <div className="py-12">
        <SectionHeading title={copy.whereTitle} description={copy.whereDescription} />
        <div className="mt-6">
          <CapabilityGrid items={copy.capabilities} />
        </div>
      </div>

      <ContactCard locale={locale} contact={linkByKey.get("support")} />
    </section>
  );
}

export function AgentRuntimeInfrastructurePage({ locale, links, projects }: ProductPageProps) {
  const linkByKey = linkMap(links);
  const agentSightImage = projectImage(projects, "agentsight");
  const copy =
    locale === "zh"
      ? {
          eyebrow: "Agent infrastructure",
          title: "Agent Runtime Infrastructure",
          description:
            "面向 OS/runtime 边界的基础设施层，用于观测、约束并保护 AI agent 工作流。",
          whyTitle: "为什么企业会关心",
          whyDescription:
            "AI agents 会执行代码、读写文件、调用工具、访问网络，并从 checkpoint 恢复。企业需要应用层日志之外的系统级 visibility 和 control。",
          stages: [
            {
              title: "Observe",
              description: "AgentSight 记录 agent 的进程、网络和工具调用行为，减少对应用层插桩的依赖。"
            },
            {
              title: "Enforce",
              description: "ActPlane 把约束下沉到 OS/eBPF 层，在 syscall、exec、file 和 network 边界做控制。"
            },
            {
              title: "Protect",
              description: "checkpoint/restore safety 关注语义回滚、恢复后的权限边界和 intent-aware fencing。"
            },
            {
              title: "Operate",
              description: "把审计、策略、事件证据和回放能力交给企业 AgentOps 或平台团队。"
            }
          ],
          componentsTitle: "Components",
          componentsDescription:
            "这些能力共同形成一个 agent runtime infrastructure 层：观测发生了什么、约束允许做什么，并保护恢复后的状态是否可信。"
        }
      : {
          eyebrow: "Agent infrastructure",
          title: "Agent Runtime Infrastructure",
          description:
            "Infrastructure for observing, enforcing, and protecting AI agent workflows at OS/runtime boundaries.",
          whyTitle: "Why it matters",
          whyDescription:
            "AI agents increasingly execute code, touch files, call tools, access networks, and resume from checkpoints. Enterprises need system-level visibility and control beyond application logs.",
          stages: [
            {
              title: "Observe",
              description: "AgentSight records process, network, and tool behavior without relying only on application-level instrumentation."
            },
            {
              title: "Enforce",
              description: "ActPlane moves control into the OS/eBPF layer across syscall, exec, file, and network boundaries."
            },
            {
              title: "Protect",
              description: "Checkpoint/restore safety covers semantic rollback risk, restored authority, and intent-aware fencing."
            },
            {
              title: "Operate",
              description: "Give AgentOps and platform teams audit trails, policy points, evidence, and replayable context."
            }
          ],
          componentsTitle: "Components",
          componentsDescription:
            "These capabilities form one agent runtime infrastructure layer: understand what happened, control what is allowed, and protect whether restored state can still be trusted."
        };

  return (
    <section className="pb-16">
      <div className="border-b border-slate-200 pb-12">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">{copy.eyebrow}</p>
        <h1 className="mt-4 max-w-4xl text-4xl font-semibold tracking-normal text-ink md:text-5xl">
          {copy.title}
        </h1>
        <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-600">{copy.description}</p>
        <div className="mt-6">
          <StarBar repos={[{ repo: "agentsight", label: "AgentSight" }]} locale={locale} />
        </div>
        <ActionRow links={[linkByKey.get("pilot")]} />
      </div>

      <div className="py-12">
        <Pipeline stages={copy.stages} />
      </div>

      <div className="grid gap-8 border-t border-slate-200 py-12 lg:grid-cols-[minmax(0,1fr)_28rem] lg:items-start">
        <SectionHeading title={copy.whyTitle} description={copy.whyDescription} />
        <div className="border border-slate-200 bg-white p-5">
          <p className="font-mono text-xs uppercase tracking-[0.16em] text-cyan-700">agent boundary</p>
          <div className="mt-5 grid grid-cols-2 gap-3 text-sm text-slate-600">
            {["code", "files", "tools", "network", "exec", "checkpoint"].map((item) => (
              <div key={item} className="border border-slate-200 px-3 py-2 font-mono text-xs">
                {item}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid gap-8 border-t border-slate-200 py-12 lg:grid-cols-[minmax(0,1fr)_28rem] lg:items-start">
        <SectionHeading title={copy.componentsTitle} description={copy.componentsDescription} />
        <VisualPanel image={agentSightImage} imageAlt="AgentSight architecture">
          <p className="font-mono text-xs uppercase tracking-[0.16em] text-cyan-700">agent runtime infra</p>
        </VisualPanel>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <article className="border border-slate-200 bg-white p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">Observability</p>
          <h3 className="mt-3 text-lg font-semibold tracking-normal text-ink">AgentSight</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            {locale === "zh"
              ? "面向 LLM 和 AI agent 的零插桩可观测性，关联进程和 TLS/网络行为。"
              : "Zero-instrumentation observability for LLM and AI agents, correlating process and TLS/network behavior."}
          </p>
          <ActionRow links={[linkByKey.get("agentsight-docs"), linkByKey.get("agentsight-github")]} />
        </article>
        <article className="border border-slate-200 bg-white p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-700">Enforcement</p>
          <h3 className="mt-3 text-lg font-semibold tracking-normal text-ink">ActPlane</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            {locale === "zh"
              ? "OS-level harness，用系统边界策略约束 agent 行为，并提供应用层日志之外的执行证据。"
              : "An OS-level harness for constraining agent behavior with system-boundary policy and execution evidence beyond application logs."}
          </p>
          <ActionRow links={[linkByKey.get("actplane-github")]} />
        </article>
        <article className="border border-slate-200 bg-white p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-amber-700">Checkpoint safety</p>
          <h3 className="mt-3 text-lg font-semibold tracking-normal text-ink">Checkpoint/restore safety</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            {locale === "zh"
              ? "基于 ACRFence 研究的安全能力，关注语义回滚、恢复后的权限边界和 intent-aware fencing。"
              : "Safety work based on ACRFence research, focused on semantic rollback risk, restored authority, and intent-aware fencing."}
          </p>
          <ActionRow links={[linkByKey.get("acrfence-article")]} />
        </article>
      </div>

      <div className="pt-12">
        <ContactCard locale={locale} contact={linkByKey.get("pilot")} />
      </div>
    </section>
  );
}

export function ServicesProductPage({ locale, links }: ProductPageProps) {
  const linkByKey = linkMap(links);
  const copy =
    locale === "zh"
      ? {
          eyebrow: "Services",
          title: "服务 / 企业支持",
          description:
            "固定范围的工程支持，面向正在采用 eBPF、bpftime 或 agent runtime infrastructure 的团队。",
          buyersTitle: "适合的团队",
          buyersDescription:
            "帮助团队评估、集成并加固 eBPF 或 agent infrastructure，从 prototype 进入 production。",
          offerings: [
            {
              label: "2 weeks",
              title: "eBPF / runtime architecture review",
              description: "评估现有系统是否适合接入 bpftime、userspace eBPF、agent observability 或 enforcement。交付架构建议、风险清单和下一步路线。"
            },
            {
              label: "4 weeks",
              title: "bpftime integration POC",
              description: "围绕一个明确场景完成 prototype、benchmark、风险清单和下一步生产化计划。"
            },
            {
              label: "Production",
              title: "Production hardening",
              description: "围绕性能、兼容性、部署、安全边界和 observability pipeline 做持续工程支持。"
            },
            {
              label: "Custom",
              title: "Custom eBPF / agent infra integration",
              description: "为特定 runtime、平台、agent workflow 或安全边界实现定制 attach path、policy point 或数据管线。"
            },
            {
              label: "Performance",
              title: "Performance tuning",
              description: "围绕 tracing overhead、JIT/AOT、事件路径、GPU instrumentation 或生产 workload 做 profiling 和 benchmark。"
            }
          ],
          buyers: [
            {
              label: "Platform",
              title: "需要把 runtime 能力接入平台的团队",
              description: "已有 tracing、profiling、sandbox 或 runtime 平台，需要更低开销或更灵活的 eBPF extension。"
            },
            {
              label: "AI infra",
              title: "需要 agent observability / enforcement 的团队",
              description: "agent 已经进入真实工作流，需要系统边界上的证据、策略和恢复安全，补足应用日志无法覆盖的行为。"
            },
            {
              label: "Research to production",
              title: "需要把原型推向生产的团队",
              description: "已有明确场景，正在补齐 benchmark、兼容性、部署模型和安全边界。"
            }
          ]
        }
      : {
          eyebrow: "Services",
          title: "Services / Enterprise Support",
          description:
            "Fixed-scope engineering support for teams adopting eBPF, bpftime, or agent runtime infrastructure.",
          buyersTitle: "Who it is for",
          buyersDescription:
            "Helps teams evaluate, integrate, and harden eBPF or agent infrastructure from prototype to production.",
          offerings: [
            {
              label: "2 weeks",
              title: "eBPF / runtime architecture review",
              description: "Assess whether a system is a good fit for bpftime, userspace eBPF, agent observability, or enforcement. Deliver architecture guidance, risk list, and next steps."
            },
            {
              label: "4 weeks",
              title: "bpftime integration POC",
              description: "Deliver a prototype, benchmark, risk list, and production plan around one sharply defined use case."
            },
            {
              label: "Production",
              title: "Production hardening",
              description: "Support performance, compatibility, deployment, security boundaries, and observability pipelines."
            },
            {
              label: "Custom",
              title: "Custom eBPF / agent infra integration",
              description: "Build custom attach paths, policy points, data pipelines, or runtime integrations for a specific platform or agent workflow."
            },
            {
              label: "Performance",
              title: "Performance tuning",
              description: "Profile and benchmark tracing overhead, JIT/AOT behavior, event paths, GPU instrumentation, and production workloads."
            }
          ],
          buyers: [
            {
              label: "Platform",
              title: "Runtime platform integration",
              description: "Tracing, profiling, sandbox, or runtime platforms that need lower-overhead or more flexible eBPF extension."
            },
            {
              label: "AI infra",
              title: "Teams adopting agent observability or enforcement",
              description: "Agent workflows need system-boundary evidence, policy, and restore safety beyond application logs."
            },
            {
              label: "Research to production",
              title: "Teams moving a prototype into production",
              description: "A clear use case exists and the next step is benchmark, compatibility, deployment model, and security boundary work."
            }
          ]
        };

  return (
    <section className="pb-16">
      <div className="border-b border-slate-200 pb-12">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">{copy.eyebrow}</p>
        <h1 className="mt-4 max-w-4xl text-4xl font-semibold tracking-normal text-ink md:text-5xl">
          {copy.title}
        </h1>
        <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-600">{copy.description}</p>
        <CredibilityStrip locale={locale} className="mt-6" />
        <div className="mt-5">
          <StarBar repos={ORG_STARS} locale={locale} />
        </div>
        <ActionRow links={[linkByKey.get("contact")]} />
      </div>

      <div className="grid gap-4 py-12 md:grid-cols-2 xl:grid-cols-3">
        {copy.offerings.map((offering) => (
          <article key={offering.title} className="border border-slate-200 bg-white p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">{offering.label}</p>
            <h2 className="mt-3 text-lg font-semibold tracking-normal text-ink">{offering.title}</h2>
            <p className="mt-3 text-sm leading-6 text-slate-600">{offering.description}</p>
          </article>
        ))}
      </div>

      <div className="border-t border-slate-200 py-12">
        <SectionHeading title={copy.buyersTitle} description={copy.buyersDescription} />
        <div className="mt-6">
          <CapabilityGrid items={copy.buyers} />
        </div>
      </div>

      <div className="grid gap-8 border-t border-slate-200 py-12 lg:grid-cols-[minmax(0,1fr)_28rem] lg:items-center">
        <SectionHeading
          title={locale === "zh" ? "交付物清晰" : "Concrete deliverables"}
          description={
            locale === "zh"
              ? "每次合作都会交付明确产物：架构建议、原型、benchmark、集成代码、部署方案或安全策略。"
              : "Every project produces concrete deliverables: architecture guidance, prototypes, benchmarks, integration code, deployment plans, or security policies."
          }
        />
        <VisualPanel>
          <p className="font-mono text-xs uppercase tracking-[0.16em] text-cyan-700">delivery loop</p>
          <div className="mt-5 space-y-3 font-mono text-xs leading-6 text-slate-700">
            <p>scope.problem()</p>
            <p>ship.prototype()</p>
            <p>harden.production()</p>
          </div>
        </VisualPanel>
      </div>

      <ContactCard locale={locale} contact={linkByKey.get("contact")} />
    </section>
  );
}
