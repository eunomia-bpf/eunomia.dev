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

function EditionsSection({ locale, contact }: { locale: Locale; contact?: ReactPageLink }) {
  const copy =
    locale === "zh"
      ? {
          eyebrow: "Open-core",
          title: "开源、独立、可持续",
          description:
            "core 是 MIT、可免费自托管且不阉割。企业版加购和支持让项目保持独立——无需 VC。",
          columns: [
            {
              name: "开源 (MIT)",
              accent: false,
              points: [
                "完整的 AgentSight + ActPlane + bpftime",
                "自托管，无功能阉割",
                "可免费用于生产",
                "通过 GitHub 获得社区支持"
              ]
            },
            {
              name: "企业版（自托管 license）",
              accent: true,
              points: [
                "SSO 与项目级 RBAC",
                "审计日志与长期留存",
                "多集群与策略管理",
                "带 SLA 的优先支持"
              ]
            },
            {
              name: "支持与赞助",
              accent: false,
              points: [
                "支持订阅",
                "Design-partner POC",
                "GitHub Sponsors 与赞助功能"
              ]
            }
          ],
          enterpriseCta: "联系我们"
        }
      : {
          eyebrow: "Open-core",
          title: "Open source, independent, and sustainable",
          description:
            "The core is MIT and free to self-host with no crippling. Commercial add-ons and support keep the project independent — no VC required.",
          columns: [
            {
              name: "Open source (MIT)",
              accent: false,
              points: [
                "Full AgentSight + ActPlane + bpftime",
                "Self-host with no feature limits",
                "Free to run in production",
                "Community support via GitHub"
              ]
            },
            {
              name: "Enterprise (self-hosted license)",
              accent: true,
              points: [
                "SSO and project-level RBAC",
                "Audit logs and long-term retention",
                "Multi-cluster and policy management",
                "Priority support with SLA"
              ]
            },
            {
              name: "Support & sponsorship",
              accent: false,
              points: [
                "Support subscriptions",
                "Design-partner POCs",
                "GitHub Sponsors and sponsored features"
              ]
            }
          ],
          enterpriseCta: "Talk to us"
        };

  return (
    <div className="border-t border-slate-200 py-12">
      <SectionHeading eyebrow={copy.eyebrow} title={copy.title} description={copy.description} />
      <div className="mt-6 grid gap-4 md:grid-cols-3">
        {copy.columns.map((column) => (
          <article
            key={column.name}
            className={[
              "rounded-lg border p-6",
              column.accent ? "border-cyan-700/30 bg-cyan-50/40" : "border-slate-200 bg-white"
            ].join(" ")}
          >
            <h3 className="text-base font-semibold tracking-normal text-ink">{column.name}</h3>
            <ul className="mt-4 space-y-2.5 text-sm leading-6 text-slate-600">
              {column.points.map((point) => (
                <li key={point} className="flex gap-2.5">
                  <span aria-hidden="true" className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-600" />
                  <span>{point}</span>
                </li>
              ))}
            </ul>
            {column.accent && contact ? (
              <a
                href={contact.href}
                className="mt-5 inline-flex min-h-10 items-center rounded-md bg-slate-950 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800"
              >
                {copy.enterpriseCta}
              </a>
            ) : null}
          </article>
        ))}
      </div>
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
          title: "AI Agent 可观测与运行管控，及其底层 eBPF 运行时",
          description:
            "旗舰方向是在系统/eBPF 边界对 AI agent 做可观测与运行管控；bpftime 是支撑它的高性能运行时引擎。开源 core 免费自托管，企业版与支持按需采用。",
          mapEyebrow: "Product map",
          mapTitle: "选择适合的工程路径",
          mapDescription:
            "从旗舰的 AI agent 可观测/管控，到底层 bpftime 运行时引擎，再到企业支持。",
          agent:
            "旗舰：在系统/eBPF 边界对 AI agent 做零插桩可观测（AgentSight）与运行时管控（ActPlane）——框架无关、约 3% 开销、内核级 ground truth。",
          bpftime:
            "底层引擎与护城河：高性能 userspace eBPF runtime，同时支撑低开销 tracing、GPU paths 和定制 runtime extension。",
          services:
            "过桥性质的 design-partner 合作：固定范围咨询、POC、生产加固、性能调优，以及 eBPF / agent infra 的定制集成。",
          buyersTitle: "适合的团队",
          buyersDescription:
            "面向在生产里运行 AI agent，并需要把开源系统工程落地的 AI infra、platform 团队。",
          flowLabels: ["AI agent 可观测/管控", "bpftime 引擎", "企业支持"],
          buyers: [
            {
              label: "AI infra / AgentOps",
              title: "在生产运行 agent",
              description: "把 agent 行为从应用日志提升到 OS/runtime 级 ground truth，支持 pilot、审计和平台集成。"
            },
            {
              label: "Platform / SRE",
              title: "低开销观测与 runtime extension",
              description: "把 uprobe、syscall、USDT、XDP 和 GPU 路径接入已有 tracing、profiling 或 runtime 平台。"
            },
            {
              label: "需要运行管控的团队",
              title: "系统边界上的 agentharness",
              description: "在进程、文件、网络、exec 和 checkpoint/restore 边界建立可审计、可执行的策略点。"
            }
          ]
        }
      : {
          eyebrow: "Products",
          title: "AI agent observability & enforcement, and the eBPF runtime underneath",
          description:
            "The flagship is system-boundary observability and runtime enforcement for AI agents; bpftime is the high-performance runtime engine that powers it. Open-source core is free to self-host, with enterprise features and support adopted as needed.",
          mapEyebrow: "Product map",
          mapTitle: "Clear engineering paths",
          mapDescription:
            "From the flagship AI agent observability & enforcement, to the bpftime runtime engine underneath, to enterprise support.",
          agent:
            "Flagship: zero-instrumentation observability (AgentSight) and runtime enforcement (ActPlane) for AI agents at the system/eBPF boundary, framework-agnostic, ~3% overhead, kernel-level ground truth.",
          bpftime:
            "The engine and moat: a high-performance userspace eBPF runtime that also powers low-overhead tracing, GPU paths, and custom runtime extension.",
          services:
            "Bridge-style design-partner work: fixed-scope consulting, POCs, production hardening, performance tuning, and custom eBPF / agent infra integration.",
          buyersTitle: "Who it helps",
          buyersDescription:
            "Built for AI infrastructure and platform teams running AI agents in production that need open-source systems engineering to land.",
          flowLabels: ["AI agent observability & enforcement", "bpftime engine", "Enterprise support"],
          buyers: [
            {
              label: "AI infra / AgentOps",
              title: "Running agents in production",
              description: "Move agent behavior beyond application logs into OS/runtime ground truth for pilots, audits, and platform integration."
            },
            {
              label: "Platform / SRE",
              title: "Low-overhead observability and runtime extension",
              description: "Connect uprobe, syscall, USDT, XDP, and GPU paths to existing tracing, profiling, or runtime platforms."
            },
            {
              label: "Teams that need runtime control",
              title: "agentharness at system boundaries",
              description: "Create auditable, enforceable policy points across process, file, network, exec, and checkpoint/restore boundaries."
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
          <ActionRow links={[linkByKey.get("agent-infra"), linkByKey.get("contact")]} />
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
            eyebrow={locale === "zh" ? "旗舰 · Agent" : "Flagship · Agent"}
            title="AI Agent Observability & Enforcement"
            description={copy.agent}
            href={linkByKey.get("agent-infra")}
            links={[linkByKey.get("agent-infra"), linkByKey.get("actplane-docs")]}
            visualLabel="agent observe + enforce"
            visualLines={["observe.agent()", "enforce.policy()", "audit.runtime()"]}
          />
          <ProductEntry
            eyebrow={locale === "zh" ? "引擎" : "Engine"}
            title="bpftime"
            description={copy.bpftime}
            href={linkByKey.get("bpftime")}
            image={bpftimeImage}
            imageAlt="bpftime"
            links={[linkByKey.get("bpftime"), linkByKey.get("bpftime-github")]}
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

      <EditionsSection locale={locale} contact={linkByKey.get("contact")} />

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
          eyebrow: "旗舰 · AI Agent 可观测与执行控制",
          title: "AI Agent 可观测与执行控制",
          description:
            "看清 AI agent 真正做了什么，并约束它能做什么，在系统边界、零插桩完成。AgentSight 负责观测，ActPlane 负责运行时执行控制。同一个 eBPF 层，位于应用之下，框架无关。",
          metrics: ["零插桩", "约 3% 开销", "框架 / 语言无关", "来自内核的 ground truth"],
          whyTitle: "为什么在系统层做",
          whyDescription:
            "应用层和 SDK tracer 只能看到 agent 自报的内容、需要插桩、带来 5–15% 开销。在应用之下的 eBPF/syscall 边界工作，拿到的是 agent 伪造不了的 ground truth，且无需改一行代码。",
          appTitle: "应用 / SDK / proxy tracer",
          appPoints: ["需要插桩或接入 proxy", "只看 agent 自报的 trace", "绑定特定框架", "5–15% 开销"],
          sysTitle: "Eunomia · 系统层",
          sysPoints: ["零插桩，开箱即用", "来自内核的 ground truth", "任意框架、任意语言", "约 3% 开销"],
          stages: [
            {
              title: "Observe 观测",
              description: "AgentSight 从系统边界记录进程、TLS/网络和工具调用行为——无 SDK、无需改代码。"
            },
            {
              title: "Enforce 执行控制",
              description: "ActPlane 在 OS/eBPF 层执行 agent 运行时策略：syscall、exec、文件和网络层的 guardrail。"
            },
            {
              title: "Protect 保护",
              description: "checkpoint/restore safety 关注语义回滚、恢复后的权限边界和 intent-aware fencing。"
            },
            {
              title: "Operate 运维",
              description: "把审计、策略、事件证据和可回放上下文交给企业 AgentOps 或平台团队。"
            }
          ],
          icpTitle: "适合谁",
          icpDescription: "在生产里真正跑 AI agent 的团队——coding agent、自治工作流、用工具的 agent。",
          icp: [
            {
              label: "AI infra / AgentOps",
              title: "在生产环境运行 agent",
              description: "需要超越应用日志的系统级证据、会话级行为关联，以及可执行的运行管控。"
            },
            {
              label: "Platform / SRE",
              title: "把 agent 接入平台",
              description: "用框架无关、低开销的方式，把 agent 行为接入已有 tracing、profiling 或 runtime 平台。"
            },
            {
              label: "跑 coding agent 的团队",
              title: "驾驭自治 agent",
              description: "为会执行代码、读写文件、调工具的 agent 设定边界，并在出问题时拿到可回放的证据。"
            }
          ],
          componentsTitle: "组成",
          componentsDescription:
            "这些能力组成同一个层：看清发生了什么、管控允许做什么，并保护恢复后的状态是否仍可信。"
        }
      : {
          eyebrow: "Flagship · AI Agent Observability & Enforcement",
          title: "AI Agent Observability & Enforcement",
          description:
            "See what your AI agents actually do, and enforce what they are allowed to do at the system boundary, with zero instrumentation. AgentSight observes; ActPlane enforces runtime policy. One eBPF layer, below the app, framework-agnostic.",
          metrics: ["Zero instrumentation", "~3% overhead", "Framework / language agnostic", "Kernel-level ground truth"],
          whyTitle: "Why the system layer",
          whyDescription:
            "App-layer and SDK tracers see only what the agent reports, need instrumentation, and add 5–15% overhead. Working below the app at the eBPF/syscall boundary gives ground truth the agent cannot forge — with no code changes.",
          appTitle: "App / SDK / proxy tracers",
          appPoints: ["Needs instrumentation or a proxy", "Sees self-reported traces", "Framework-specific", "5–15% overhead"],
          sysTitle: "Eunomia · system layer",
          sysPoints: ["Zero instrumentation, drop-in", "Ground truth from the kernel", "Any framework, any language", "~3% overhead"],
          stages: [
            {
              title: "Observe",
              description: "AgentSight records process, TLS/network, and tool behavior from the system boundary — no SDK, no code changes."
            },
            {
              title: "Enforce",
              description: "ActPlane enforces what agents can do at the OS/eBPF layer: syscall, exec, file, and network guardrails."
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
          icpTitle: "Who it's for",
          icpDescription: "Teams running real AI agents in production — coding agents, autonomous workflows, and tool-using agents.",
          icp: [
            {
              label: "AI infra / AgentOps",
              title: "Running agents in production",
              description: "Need system-level evidence beyond app logs, session-level behavior correlation, and enforceable runtime control."
            },
            {
              label: "Platform / SRE",
              title: "Bringing agents onto the platform",
              description: "Connect agent behavior to existing tracing, profiling, or runtime platforms in a framework-agnostic, low-overhead way."
            },
            {
              label: "Teams shipping coding agents",
              title: "Steering autonomous agents",
              description: "Set boundaries for agents that execute code, touch files, and call tools — with replayable evidence when something goes wrong."
            }
          ],
          componentsTitle: "Components",
          componentsDescription:
            "These capabilities form one layer: understand what happened, control what is allowed, and protect whether restored state can still be trusted."
        };

  return (
    <section className="pb-16">
      <div className="border-b border-slate-200 pb-12">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">{copy.eyebrow}</p>
        <h1 className="mt-4 max-w-4xl text-4xl font-semibold tracking-normal text-ink md:text-5xl">
          {copy.title}
        </h1>
        <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-600">{copy.description}</p>
        <div className="mt-6 flex flex-wrap gap-2">
          {copy.metrics.map((metric) => (
            <span
              key={metric}
              className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600"
            >
              <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-cyan-600" />
              {metric}
            </span>
          ))}
        </div>
        <div className="mt-5">
          <StarBar repos={[{ repo: "agentsight", label: "AgentSight" }]} locale={locale} />
        </div>
        <ActionRow links={[linkByKey.get("pilot")]} />
      </div>

      <div className="py-12">
        <Pipeline stages={copy.stages} />
      </div>

      <div className="border-t border-slate-200 py-12">
        <SectionHeading title={copy.whyTitle} description={copy.whyDescription} />
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <article className="rounded-lg border border-slate-200 bg-white p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">{copy.appTitle}</p>
            <ul className="mt-4 space-y-2.5 text-sm leading-6 text-slate-500">
              {copy.appPoints.map((point) => (
                <li key={point} className="flex gap-2.5">
                  <span aria-hidden="true" className="mt-2 h-1 w-3 shrink-0 rounded-full bg-slate-300" />
                  <span>{point}</span>
                </li>
              ))}
            </ul>
          </article>
          <article className="rounded-lg border border-cyan-700/30 bg-cyan-50/40 p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">{copy.sysTitle}</p>
            <ul className="mt-4 space-y-2.5 text-sm leading-6 text-slate-700">
              {copy.sysPoints.map((point) => (
                <li key={point} className="flex gap-2.5">
                  <span aria-hidden="true" className="mt-1.5 shrink-0 text-cyan-600">✓</span>
                  <span>{point}</span>
                </li>
              ))}
            </ul>
          </article>
        </div>
      </div>

      <div className="border-t border-slate-200 py-12">
        <SectionHeading title={copy.icpTitle} description={copy.icpDescription} />
        <div className="mt-6">
          <CapabilityGrid items={copy.icp} />
        </div>
      </div>

      <div className="grid gap-8 border-t border-slate-200 py-12 lg:grid-cols-[minmax(0,1fr)_28rem] lg:items-start">
        <SectionHeading title={copy.componentsTitle} description={copy.componentsDescription} />
        <VisualPanel image={agentSightImage} imageAlt="AgentSight architecture">
          <p className="font-mono text-xs uppercase tracking-[0.16em] text-cyan-700">agent runtime infra</p>
        </VisualPanel>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <article className="border border-slate-200 bg-white p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">Observe &amp; safeguard</p>
          <h3 className="mt-3 text-lg font-semibold tracking-normal text-ink">AgentSight</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            {locale === "zh"
              ? "面向 LLM 和 AI agent 的零插桩可观测性，关联进程与 TLS/网络行为；同时涵盖 checkpoint/restore safety（ACRFence）与资源/性能控制（agentcgroup）。"
              : "Zero-instrumentation observability for LLM and AI agents — correlate process and TLS/network behavior. Also covers checkpoint/restore safety (ACRFence) and resource/performance control (agentcgroup)."}
          </p>
          <ActionRow
            links={[linkByKey.get("agentsight-docs"), linkByKey.get("agentsight-github"), linkByKey.get("acrfence-article")]}
          />
        </article>
        <article className="border border-slate-200 bg-white p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-700">Harness</p>
          <h3 className="mt-3 text-lg font-semibold tracking-normal text-ink">ActPlane</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            {locale === "zh"
              ? "OS-level harness，在 syscall、exec、file 和 network 边界用系统级策略约束 agent 行为，并提供应用层日志之外的执行证据。"
              : "An OS-level harness that shapes agent behavior with system-boundary policy across syscall, exec, file, and network — plus execution evidence beyond application logs."}
          </p>
          <ActionRow links={[linkByKey.get("actplane-docs"), linkByKey.get("actplane-github")]} />
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
            "以 design-partner 方式合作：支持订阅、固定范围的 POC 与生产加固，帮助采用 eBPF、bpftime 或 AI agent 基础设施的团队从原型走到生产。开源 core 始终免费——这里是需要 SLA、定制集成或深度协作时的入口。",
          buyersTitle: "适合的团队",
          buyersDescription:
            "帮助团队评估、集成并加固 eBPF 或 agent infrastructure，从 prototype 进入 production。",
          offerings: [
            {
              label: "Subscription",
              title: "Support subscription",
              description: "带 SLA 的持续支持：升级、答疑、生产事故协助，以及对 enterprise license 功能的支持。"
            },
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
            "Design-partner engagements and support: subscriptions, fixed-scope POCs, and production hardening for teams adopting eBPF, bpftime, or AI agent infrastructure. The open-source core is always free — this is the way in when you need an SLA, custom integration, or deep collaboration.",
          buyersTitle: "Who it is for",
          buyersDescription:
            "Helps teams evaluate, integrate, and harden eBPF or agent infrastructure from prototype to production.",
          offerings: [
            {
              label: "Subscription",
              title: "Support subscription",
              description: "Ongoing support with an SLA: upgrades, troubleshooting, production incident help, and support for enterprise license features."
            },
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
