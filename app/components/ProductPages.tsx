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

function ProductFigure({
  image,
  imageAlt,
  caption,
  className = ""
}: {
  image: string;
  imageAlt: string;
  caption: string;
  className?: string;
}) {
  return (
    <figure className={`overflow-hidden border border-slate-200 bg-white ${className}`.trim()}>
      <div className="relative aspect-video bg-slate-950">
        <Image
          src={image}
          alt={imageAlt}
          fill
          sizes="(min-width: 1024px) 38rem, 100vw"
          className="object-contain"
          unoptimized
        />
      </div>
      <figcaption className="border-t border-slate-200 px-4 py-3 text-sm leading-6 text-slate-600">
        {caption}
      </figcaption>
    </figure>
  );
}

function EditionsSection({ locale, contact }: { locale: Locale; contact?: ReactPageLink }) {
  const copy =
    locale === "zh"
      ? {
          eyebrow: "Open-core",
          title: "开源、企业就绪、可规模化",
          description:
            "core 是 MIT、可免费自托管；需要生产集成和 SLA 的团队可按需采用商业功能与支持。",
          columns: [
            {
              name: "开源 (MIT)",
              accent: false,
              points: [
                "完整的 AgentSight + ActPlane + Akeep + bpftime",
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
              name: "企业服务",
              accent: false,
              points: [
                "生产集成与部署支持",
                "Design-partner POC",
                "带 SLA 的优先工程支持"
              ]
            }
          ],
          enterpriseCta: "联系我们"
        }
      : {
          eyebrow: "Open-core",
          title: "Open source, enterprise-ready, and built to scale",
          description:
            "The core is MIT and free to self-host; commercial add-ons and support are available for teams that need production integration and SLAs.",
          columns: [
            {
              name: "Open source (MIT)",
              accent: false,
              points: [
                "Full AgentSight + ActPlane + Akeep + bpftime",
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
              name: "Enterprise services",
              accent: false,
              points: [
                "Production integration and deployment",
                "Design-partner POCs",
                "Priority engineering support with SLA"
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
          title: "AI Agent 系统层，及其底层 eBPF 运行时",
          description:
            "旗舰方向是在系统层观察、约束并保留 AI Agent 的长期执行过程；bpftime 是支撑低开销系统观测与扩展的运行时引擎。开源 core 免费自托管，企业版与支持按需采用。",
          mapEyebrow: "Product map",
          mapTitle: "选择适合的工程路径",
          mapDescription:
            "从 AgentSight、ActPlane 与 Akeep 组成的系统层，到底层 bpftime 运行时引擎，再到企业支持。",
          agent:
            "旗舰：AgentSight 观察和解释执行过程，ActPlane 在系统边界执行策略，Akeep 保存可检查、可恢复的原生会话历史。",
          bpftime:
            "底层引擎与护城河：高性能 userspace eBPF runtime，同时支撑低开销 tracing、GPU paths 和定制 runtime extension。",
          services:
            "过桥性质的 design-partner 合作：固定范围咨询、POC、生产加固、性能调优，以及 eBPF / agent infra 的定制集成。",
          buyersTitle: "适合的团队",
          buyersDescription:
            "面向在生产里运行 AI agent，并需要把开源系统工程落地的 AI infra、platform 团队。",
          flowLabels: ["AI Agent 系统层", "bpftime 引擎", "企业支持"],
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
              title: "系统边界上的执行平面",
              description: "在进程、文件、网络和 exec 边界建立可审计的策略点，并保留可检查、可恢复的会话历史。"
            }
          ]
        }
      : {
          eyebrow: "Products",
          title: "The AI agent system layer, and the eBPF runtime underneath",
          description:
            "The flagship observes, governs, and preserves long-running AI agent execution at the system layer; bpftime is the runtime engine underneath. The open-source core is free to self-host, with enterprise features and support adopted as needed.",
          mapEyebrow: "Product map",
          mapTitle: "Clear engineering paths",
          mapDescription:
            "From the AgentSight, ActPlane, and Akeep system layer, to the bpftime runtime engine underneath, to enterprise support.",
          agent:
            "Flagship: AgentSight observes and explains execution, ActPlane enforces policy at system boundaries, and Akeep preserves inspectable, recoverable native session history.",
          bpftime:
            "The engine and moat: a high-performance userspace eBPF runtime that also powers low-overhead tracing, GPU paths, and custom runtime extension.",
          services:
            "Bridge-style design-partner work: fixed-scope consulting, POCs, production hardening, performance tuning, and custom eBPF / agent infra integration.",
          buyersTitle: "Who it helps",
          buyersDescription:
            "Built for AI infrastructure and platform teams running AI agents in production that need open-source systems engineering to land.",
          flowLabels: ["AI agent system layer", "bpftime engine", "Enterprise support"],
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
              title: "An execution plane at system boundaries",
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
            title="Agent System Layer"
            description={copy.agent}
            href={linkByKey.get("agent-infra")}
            links={[linkByKey.get("agent-infra"), linkByKey.get("presentation")]}
            visualLabel="observe + enforce + recover"
            visualLines={["observe.agent()", "enforce.policy()", "recover.session()"]}
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
  const agentNebulaImage = "/_content-assets/docs/presentations/agent-system-layer/images/agent-nebula.png";
  const agentSightTopImage = "/_content-assets/docs/agentsight/images/top-mode-demo.png";
  const agentSightFlamegraphImage =
    "/_content-assets/docs/agentsight/flamegraph-example/semantic-flamegraph-top200.svg";
  const actPlaneImage = "/_content-assets/docs/presentations/agent-system-layer/images/actplane-policy-dsl.png";
  const actPlaneRuntimeImage =
    "/_content-assets/docs/presentations/agent-system-layer/images/actplane-runtime-loop.png";
  const copy =
    locale === "zh"
      ? {
          eyebrow: "Agent System Layer",
          title: "AI Agent 的操作系统级执行平面",
          description:
            "长期运行的 AI Agent 会跨越模型、工具、进程、文件和网络。AgentSight 观察并解释这些执行活动，ActPlane 把策略落实到操作系统边界，Akeep 保存 provider-native 会话历史，让几天或几周的工作仍然可以检查和恢复。",
          metrics: ["适配现有 Agent", "操作系统级证据", "内核路径执行控制", "可恢复的会话历史"],
          whyTitle: "长期运行的 Agent 需要应用 trace 之外的系统层",
          whyDescription:
            "应用层工具适合观测 prompt、token、eval 和 latency。系统层继续回答实际执行了什么、策略是否真正生效，以及工作中断后能否从可信历史恢复。",
          appTitle: "应用层 / SDK / gateway",
          appPoints: [
            "每个应用需要接入 SDK、callback 或 gateway",
            "closed-source CLI 只能依赖它主动暴露的日志",
            "trace 通常停在 framework 或 process 边界",
            "容易漏掉 subprocess 和本地文件活动"
          ],
          sysTitle: "Agent system layer",
          sysPoints: [
            "AgentSight 从进程外部关联 LLM、process、file 和 network 事件",
            "ActPlane 在 exec、file、network 和 syscall 路径执行策略",
            "Akeep 保存 provider-native 历史，并支持完整性检查与恢复",
            "三部分可以独立使用，也可以组成同一个执行平面"
          ],
          stages: [
            {
              title: "观察",
              description: "查看活跃 session、进程、模型与工具调用，以及文件、网络和资源活动。"
            },
            {
              title: "解释",
              description: "把 prompt、skill 和工具决策关联到系统效果，理解注意力、迭代与失败路径。"
            },
            {
              title: "执行控制",
              description: "把高层意图和时序上下文编译成操作系统边界上的 allow、block、kill 或 notify。"
            },
            {
              title: "保存与恢复",
              description: "版本化保存原生会话文件，检查历史完整性，并在设备或 provider 变化后继续工作。"
            }
          ],
          icpTitle: "这个系统层回答什么问题",
          icpDescription: "从长期运行、平台接入和安全治理中的真实问题出发。",
          icp: [
            {
              label: "几天或几周的自主工作",
              title: "30 秒理解 Agent 做了什么",
              description: "看见注意力如何移动、哪些文件和模块真正变化、测试与实现如何交替，以及哪些尝试反复失败。"
            },
            {
              label: "Platform / SRE",
              title: "接入现有 observability 平台",
              description: "无需逐个修改 agent，把 closed-source CLI 和不同框架的系统行为导出到已有 tracing 或 profiling 管线。"
            },
            {
              label: "Security / governance",
              title: "审计、约束并恢复系统影响",
              description: "用 AgentSight 取得执行证据，由 ActPlane 强制策略，再用 Akeep 保留可验证、可恢复的会话历史。"
            }
          ],
          componentsTitle: "看看系统层实际呈现什么",
          componentsDescription:
            "这些画面来自项目现有界面和演示：AgentSight 把长期执行变成可观察的结构，ActPlane 把策略落实到执行路径，Akeep 保留可以检查和恢复的原生历史。",
          agentSightVisuals: {
            title: "AgentSight · 观察与解释",
            description: "从当前正在运行的 Agent，到一周内注意力如何移动，再到时间和工具调用花在哪里。",
            nebula: "AgentNebula 按真实读写时间回放文件结构与注意力的变化。",
            top: "实时 top 视图汇总 session、模型、token、健康状态、进程、工具和文件活动。",
            flamegraph: "语义火焰图把 prompt 与工具路径连接到带权重的系统效果。",
            architecture: "AgentSight 从系统事件中关联 Agent、模型调用与执行行为。"
          },
          actPlaneVisuals: {
            title: "ActPlane · 在执行路径上强制策略",
            description: "高层规则被编译成系统策略；解释留在用户态，强制执行落在进程必须经过的内核路径。",
            policy: "策略 DSL 保留意图和时序上下文，并在操作系统边界执行。",
            runtime: "运行闭环连接策略、编译器、权限检查、IFC 引擎与语义反馈。"
          },
          akeepVisuals: {
            title: "Akeep · 保留与恢复原生会话历史",
            description: "Akeep 直接版本化 provider-native 文件；普通流程可以只用 commit，需要时再做 diff、完整性检查和恢复。",
            boundary: "设备或 provider 发生变化后，历史仍可检查、验证并恢复。"
          }
        }
      : {
          eyebrow: "Agent System Layer",
          title: "The OS-level execution plane for AI agents",
          description:
            "Long-running AI agents cross models, tools, processes, files, and networks. AgentSight observes and explains that execution, ActPlane turns policy into enforcement at operating-system boundaries, and Akeep preserves provider-native session history so days or weeks of work remain inspectable and recoverable.",
          metrics: ["Works with existing agents", "OS-level evidence", "Enforcement on the execution path", "Recoverable session history"],
          whyTitle: "Long-running agents need a system layer beyond application traces",
          whyDescription:
            "Application-level tools explain prompts, tokens, evals, and latency. The system layer continues the story: what actually executed, whether policy held on the execution path, and whether interrupted work can be recovered from trustworthy history.",
          appTitle: "Application / SDK / gateway",
          appPoints: [
            "Each application needs an SDK, callback, or gateway integration",
            "Closed-source CLIs are limited to the logs they expose",
            "Traces often stop at framework or process boundaries",
            "Subprocess and local file activity can be missed"
          ],
          sysTitle: "Agent system layer",
          sysPoints: [
            "AgentSight correlates LLM, process, file, and network events from outside the process",
            "ActPlane enforces policy across exec, file, network, and syscall paths",
            "Akeep preserves provider-native history with integrity checks and recovery",
            "Adopt each component independently or use them as one execution plane"
          ],
          stages: [
            {
              title: "Observe",
              description: "Inspect active sessions, processes, model and tool calls, file and network activity, and resource use."
            },
            {
              title: "Explain",
              description: "Connect prompts, skills, and tool decisions to system effects, attention shifts, iteration paths, and failures."
            },
            {
              title: "Enforce",
              description: "Compile intent and temporal context into allow, block, kill, or notify decisions at operating-system boundaries."
            },
            {
              title: "Preserve & recover",
              description: "Version provider-native session files, verify history integrity, and continue after a device or provider changes."
            }
          ],
          icpTitle: "What the system layer should answer",
          icpDescription: "Start from real problems in long-running work, platform integration, and governance.",
          icp: [
            {
              label: "Days or weeks of autonomous work",
              title: "Understand the run in 30 seconds",
              description: "See where attention moved, which files and modules changed, how tests and implementation alternated, and where attempts repeatedly failed."
            },
            {
              label: "Platform / SRE",
              title: "Connect existing observability",
              description: "Export system behavior from closed-source CLIs and mixed agent frameworks into existing tracing or profiling pipelines without modifying each agent."
            },
            {
              label: "Security / governance",
              title: "Audit, govern, and recover system effects",
              description: "Use AgentSight for execution evidence, ActPlane for enforced policy, and Akeep for verifiable, recoverable session history."
            }
          ],
          componentsTitle: "See what the system layer actually produces",
          componentsDescription:
            "These are existing project interfaces and presentation visuals: AgentSight makes long-running execution observable, ActPlane moves policy onto the execution path, and Akeep preserves native history for inspection and recovery.",
          agentSightVisuals: {
            title: "AgentSight · Observe and explain",
            description: "Move from agents running now, to attention moving across a week, to where time and tool calls accumulated.",
            nebula: "AgentNebula replays file structure and attention using the agent's actual read and write timing.",
            top: "The live top view summarizes sessions, models, tokens, health, processes, tools, and file activity.",
            flamegraph: "The semantic flamegraph connects prompts and tool paths to weighted system effects.",
            architecture: "AgentSight correlates agents, model calls, and execution behavior from system events."
          },
          actPlaneVisuals: {
            title: "ActPlane · Enforce on the execution path",
            description: "High-level rules compile into system policy; interpretation stays in userspace while enforcement sits on kernel paths every process must cross.",
            policy: "The policy DSL carries intent and temporal context into operating-system enforcement.",
            runtime: "The runtime loop connects policy, compilation, authority checks, the IFC engine, and semantic feedback."
          },
          akeepVisuals: {
            title: "Akeep · Preserve and recover native session history",
            description: "Akeep versions provider-native files directly. Ordinary use can be a commit; diff, integrity checks, and recovery are available when needed.",
            boundary: "When a device or provider changes, the history remains inspectable, verifiable, and recoverable."
          }
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
          <StarBar
            repos={[
              { repo: "agentsight", label: "AgentSight" },
              { repo: "ActPlane", label: "ActPlane" },
              { repo: "akeep", label: "Akeep" }
            ]}
            locale={locale}
          />
        </div>
        <ActionRow links={[linkByKey.get("agentsight-docs"), linkByKey.get("presentation"), linkByKey.get("pilot")]} />
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

      <div className="border-t border-slate-200 py-12">
        <SectionHeading title={copy.componentsTitle} description={copy.componentsDescription} />
        <div className="mt-10">
          <h3 className="text-xl font-semibold tracking-normal text-ink">{copy.agentSightVisuals.title}</h3>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">{copy.agentSightVisuals.description}</p>
          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <ProductFigure
              image={agentNebulaImage}
              imageAlt="AgentNebula replay of AI agent file activity"
              caption={copy.agentSightVisuals.nebula}
              className="lg:col-span-2"
            />
            <ProductFigure
              image={agentSightTopImage}
              imageAlt="AgentSight top view showing live AI agent sessions"
              caption={copy.agentSightVisuals.top}
            />
            <ProductFigure
              image={agentSightFlamegraphImage}
              imageAlt="AgentSight semantic flamegraph of AI agent activity"
              caption={copy.agentSightVisuals.flamegraph}
            />
            {agentSightImage ? (
              <ProductFigure
                image={agentSightImage}
                imageAlt="AgentSight architecture"
                caption={copy.agentSightVisuals.architecture}
                className="lg:col-span-2"
              />
            ) : null}
          </div>
        </div>

        <div className="mt-12 border-t border-slate-200 pt-10">
          <h3 className="text-xl font-semibold tracking-normal text-ink">{copy.actPlaneVisuals.title}</h3>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">{copy.actPlaneVisuals.description}</p>
          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <ProductFigure
              image={actPlaneImage}
              imageAlt="ActPlane policy DSL compiled into operating-system enforcement"
              caption={copy.actPlaneVisuals.policy}
            />
            <ProductFigure
              image={actPlaneRuntimeImage}
              imageAlt="ActPlane runtime architecture with policy, compiler, IFC engine, and feedback"
              caption={copy.actPlaneVisuals.runtime}
            />
          </div>
        </div>

        <div className="mt-12 border-t border-slate-200 pt-10">
          <h3 className="text-xl font-semibold tracking-normal text-ink">{copy.akeepVisuals.title}</h3>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">{copy.akeepVisuals.description}</p>
          <div className="mt-6 border border-slate-800 bg-slate-950 p-6 font-mono text-sm leading-8 text-slate-300 sm:p-8">
            <p><span className="text-emerald-300">Day 1</span> &nbsp; akeep commit -m &quot;first working path&quot;</p>
            <p><span className="text-emerald-300">Day 2</span> &nbsp; akeep commit -m &quot;after failed migration&quot;</p>
            <p><span className="text-emerald-300">Day 7</span> &nbsp; akeep diff HEAD~1 HEAD</p>
            <p className="mt-5 border-t border-slate-700 pt-5 text-slate-400">{copy.akeepVisuals.boundary}</p>
            <p className="mt-5 text-white">akeep fsck HEAD</p>
            <p className="text-white">akeep checkout HEAD --to /tmp/recovery</p>
            <p className="mt-5 text-emerald-300">Continue.</p>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <article className="border border-slate-200 bg-white p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">Observe &amp; explain</p>
          <h3 className="mt-3 text-lg font-semibold tracking-normal text-ink">AgentSight</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            {locale === "zh"
              ? "本地优先的 AI agent profiling 与 monitoring，把 prompt、模型和工具决策关联到进程、文件、网络与资源行为，并支持 report、agentpprof 和 OpenTelemetry 导出。"
              : "Local-first AI agent profiling and monitoring that connects prompts, models, and tool decisions to process, file, network, and resource activity, with reports, agentpprof, and OpenTelemetry export."}
          </p>
          <ActionRow
            links={[linkByKey.get("agentsight-docs"), linkByKey.get("agentsight-github")]}
          />
        </article>
        <article className="border border-slate-200 bg-white p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-rose-700">Enforce</p>
          <h3 className="mt-3 text-lg font-semibold tracking-normal text-ink">ActPlane</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            {locale === "zh"
              ? "把高层意图与时序上下文编译成系统策略，在 syscall、exec、file 和 network 路径执行 allow、block、kill 或 notify。"
              : "Compiles intent and temporal context into policy enforced across syscall, exec, file, and network paths, with concrete feedback when an action is denied."}
          </p>
          <ActionRow links={[linkByKey.get("actplane-docs"), linkByKey.get("actplane-github")]} />
        </article>
        <article className="border border-slate-200 bg-white p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-700">Preserve &amp; recover</p>
          <h3 className="mt-3 text-lg font-semibold tracking-normal text-ink">Akeep</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            {locale === "zh"
              ? "版本化保存 Agent 的 provider-native 会话文件，支持完整性检查、差异比较与恢复，无需把历史转换成另一套 memory 格式。"
              : "Versions provider-native agent session files for integrity checks, diffs, and recovery without converting history into another agent-memory format."}
          </p>
          <ActionRow links={[linkByKey.get("akeep-github")]} />
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
