import Image from "next/image";

import type { BlogEntry } from "../lib/content/types";
import { localizePath } from "../lib/paths";
import type { Locale } from "../lib/site-data";
import { BlogPostList } from "./BlogPostList";

type HomePageLandingProps = {
  locale: Locale;
  recentPosts: BlogEntry[];
};

type HomeCopy = {
  kicker: string;
  title: string;
  summary: string;
  primaryCta: string;
  secondaryCta: string;
  githubCta: string;
  imageAlt: string;
  proof: Array<{ value: string; label: string }>;
  sectionLabel: string;
  sectionTitle: string;
  sectionIntro: string;
  featured: Array<{ label: string; title: string; description: string; href: string }>;
  projectTitle: string;
  projectIntro: string;
  projects: Array<{ title: string; description: string; href: string; tag: string }>;
  latestTitle: string;
  allPosts: string;
};

const copyByLocale: Record<Locale, HomeCopy> = {
  en: {
    kicker: "Open systems research and tooling",
    title: "Eunomia",
    summary:
      "An open-source lab for eBPF systems: runnable tutorials, userspace runtimes, LLM-assisted tracing, and security research for agent workflows.",
    primaryCta: "Start tutorials",
    secondaryCta: "Read research",
    githubCta: "GitHub",
    imageAlt: "eBPF architecture and runtime map",
    proof: [
      { value: "OSDI 2025", label: "bpftime userspace eBPF runtime" },
      { value: "eBPF 2024", label: "LLM-assisted kernel tracing" },
      { value: "arXiv 2603.20625", label: "agent checkpoint/restore safety" }
    ],
    sectionLabel: "Work",
    sectionTitle: "Research ideas that keep shipping as tools.",
    sectionIntro:
      "The site connects papers, project documentation, and executable examples without moving readers away from the stable docs URLs.",
    featured: [
      {
        label: "Runtime",
        title: "bpftime",
        description:
          "A high-performance userspace eBPF runtime for uprobes, USDT, syscall, XDP, GPU, and extension workflows.",
        href: "/bpftime/"
      },
      {
        label: "Research",
        title: "ACRFence",
        description:
          "Published work on semantic rollback attacks in agent checkpoint/restore and intent-aware fencing.",
        href: "https://arxiv.org/abs/2603.20625"
      },
      {
        label: "AI tracing",
        title: "GPTtrace",
        description:
          "Generate eBPF tracing programs with natural language and use LLMs to explore Linux kernel behavior.",
        href: "/GPTtrace/"
      }
    ],
    projectTitle: "Project paths",
    projectIntro: "Stable entry points for the documentation and research areas behind eunomia-bpf.",
    projects: [
      {
        title: "Tutorials",
        description: "Executable eBPF walkthroughs from hello world to sched-ext, networking, GPU, and userspace topics.",
        href: "/tutorials/",
        tag: "Learn"
      },
      {
        title: "eunomia-bpf",
        description: "Build, package, distribute, and run eBPF programs with JSON metadata and OCI workflows.",
        href: "/eunomia-bpf/",
        tag: "Toolchain"
      },
      {
        title: "llvmbpf",
        description: "LLVM JIT/AOT support for userspace eBPF execution, used as the compiler and runtime core of bpftime.",
        href: "/bpftime/llvmbpf/",
        tag: "Runtime"
      },
      {
        title: "Ecosystem",
        description: "Papers, talks, references, and adjacent work across eBPF and systems tooling.",
        href: "/others/",
        tag: "Research"
      }
    ],
    latestTitle: "Latest writing",
    allPosts: "All posts"
  },
  zh: {
    kicker: "开源系统研究与工具",
    title: "Eunomia",
    summary:
      "Eunomia 是围绕 eBPF 系统的开源实验室：可运行教程、userspace runtime、LLM tracing，以及 AI agent 安全研究。",
    primaryCta: "从教程开始",
    secondaryCta: "阅读研究",
    githubCta: "GitHub",
    imageAlt: "eBPF 架构与运行时路线图",
    proof: [
      { value: "OSDI 2025", label: "bpftime userspace eBPF runtime" },
      { value: "eBPF 2024", label: "LLM 辅助内核追踪" },
      { value: "arXiv 2603.20625", label: "Agent checkpoint/restore 安全" }
    ],
    sectionLabel: "方向",
    sectionTitle: "研究想法会继续落到可用工具里。",
    sectionIntro:
      "这个站点把论文、项目文档和可执行示例连在一起，同时保留读者已经使用的稳定文档 URL。",
    featured: [
      {
        label: "Runtime",
        title: "bpftime",
        description:
          "面向 userspace 的高性能 eBPF runtime，覆盖 uprobe、USDT、syscall、XDP、GPU 和扩展工作流。",
        href: "/bpftime/"
      },
      {
        label: "Research",
        title: "ACRFence",
        description:
          "已经发布的 agent checkpoint/restore 语义回滚攻击与 intent-aware fencing 研究。",
        href: "https://arxiv.org/abs/2603.20625"
      },
      {
        label: "AI tracing",
        title: "GPTtrace",
        description:
          "用自然语言生成 eBPF tracing 程序，并借助 LLM 探索 Linux 内核行为。",
        href: "/GPTtrace/"
      }
    ],
    projectTitle: "项目入口",
    projectIntro: "eunomia-bpf 相关文档和研究方向的稳定入口。",
    projects: [
      {
        title: "教程",
        description: "从 hello world 到 sched-ext、网络、GPU 和 userspace 主题的可执行 eBPF 示例。",
        href: "/tutorials/",
        tag: "Learn"
      },
      {
        title: "eunomia-bpf",
        description: "通过 JSON metadata 和 OCI workflow 构建、打包、分发并运行 eBPF 程序。",
        href: "/eunomia-bpf/",
        tag: "Toolchain"
      },
      {
        title: "llvmbpf",
        description: "支持 LLVM JIT/AOT 的 userspace eBPF 执行能力，也是 bpftime 的编译和运行核心。",
        href: "/bpftime/llvmbpf/",
        tag: "Runtime"
      },
      {
        title: "生态",
        description: "eBPF 与系统工具方向的论文、演讲、参考资料和相关工作。",
        href: "/others/",
        tag: "Research"
      }
    ],
    latestTitle: "最新文章",
    allPosts: "全部文章"
  }
};

function localizedHref(href: string, locale: Locale): string {
  return href.startsWith("/") ? localizePath(href, locale) : href;
}

export function HomePageHero({ locale }: { locale: Locale }) {
  const copy = copyByLocale[locale];

  return (
    <section className="relative isolate overflow-hidden border-b border-slate-200 bg-slate-950 text-white">
      <Image
        src="/ebpf_arch.png"
        alt=""
        aria-hidden="true"
        fill
        priority
        sizes="100vw"
        className="object-cover opacity-20"
        unoptimized
      />
      <div className="absolute inset-0 bg-slate-950/75" />
      <div className="relative mx-auto max-w-[82rem] px-4 py-14 sm:px-6 lg:px-8 lg:py-20">
        <div className="max-w-4xl">
          <p className="text-xs font-semibold uppercase tracking-normal text-cyan-200">{copy.kicker}</p>
          <h1 className="mt-4 text-4xl font-semibold tracking-normal text-white sm:text-5xl lg:text-6xl">
            {copy.title}
          </h1>
          <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-200">{copy.summary}</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href={localizedHref("/tutorials/", locale)}
              className="inline-flex min-h-11 items-center rounded-lg bg-white px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-100"
            >
              {copy.primaryCta}
            </a>
            <a
              href={localizedHref("/blog/", locale)}
              className="inline-flex min-h-11 items-center rounded-lg border border-white/35 px-5 py-2.5 text-sm font-semibold text-white transition hover:border-white hover:bg-white/10"
            >
              {copy.secondaryCta}
            </a>
            <a
              href="https://github.com/eunomia-bpf/"
              className="inline-flex min-h-11 items-center rounded-lg border border-white/20 px-5 py-2.5 text-sm font-semibold text-slate-100 transition hover:border-white/60 hover:bg-white/10"
            >
              {copy.githubCta}
            </a>
          </div>
        </div>
        <dl className="mt-12 grid gap-3 sm:grid-cols-3">
          {copy.proof.map((item) => (
            <div key={item.value} className="border-l border-cyan-300/60 pl-4">
              <dt className="text-sm font-semibold text-white">{item.value}</dt>
              <dd className="mt-1 text-sm leading-5 text-slate-300">{item.label}</dd>
            </div>
          ))}
        </dl>
        <span className="sr-only">{copy.imageAlt}</span>
      </div>
    </section>
  );
}

export function HomePageLanding({ locale, recentPosts }: HomePageLandingProps) {
  const copy = copyByLocale[locale];

  return (
    <div className="pb-16">
      <section className="grid gap-10 border-b border-slate-200 pb-12 lg:grid-cols-[minmax(0,0.95fr)_minmax(22rem,1.05fr)] lg:items-start">
        <div>
          <p className="text-xs font-semibold uppercase tracking-normal text-cyan-700">{copy.sectionLabel}</p>
          <h2 className="mt-3 max-w-2xl text-3xl font-semibold tracking-normal text-ink md:text-4xl">
            {copy.sectionTitle}
          </h2>
          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">{copy.sectionIntro}</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
          {copy.featured.map((item) => (
            <a
              key={item.title}
              href={localizedHref(item.href, locale)}
              className="group rounded-lg border border-slate-200 bg-white p-5 transition hover:border-cyan-700/40 hover:bg-slate-50"
            >
              <p className="text-xs font-semibold uppercase tracking-normal text-cyan-700">{item.label}</p>
              <h3 className="mt-3 text-lg font-semibold tracking-normal text-ink group-hover:text-cyan-800">
                {item.title}
              </h3>
              <p className="mt-2 text-sm leading-6 text-slate-600">{item.description}</p>
            </a>
          ))}
        </div>
      </section>

      <section className="py-12" aria-labelledby="home-project-paths">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 id="home-project-paths" className="text-2xl font-semibold tracking-normal text-ink">
              {copy.projectTitle}
            </h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">{copy.projectIntro}</p>
          </div>
          <Image
            src="/llvmbpf.png"
            alt="llvmbpf"
            width={112}
            height={64}
            className="hidden h-16 w-28 rounded-lg border border-slate-200 bg-slate-50 object-contain p-2 sm:block"
            unoptimized
          />
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {copy.projects.map((project) => (
            <a
              key={project.href}
              href={localizedHref(project.href, locale)}
              className="group rounded-lg border border-slate-200 bg-white p-5 transition hover:border-slate-300 hover:bg-slate-50"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="text-xs font-semibold uppercase tracking-normal text-slate-500">{project.tag}</p>
                  <h3 className="mt-2 text-lg font-semibold tracking-normal text-ink group-hover:text-cyan-800">
                    {project.title}
                  </h3>
                </div>
                <span
                  aria-hidden="true"
                  className="mt-1 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-slate-200 text-lg text-slate-500 group-hover:border-cyan-700/30 group-hover:text-cyan-800"
                >
                  +
                </span>
              </div>
              <p className="mt-3 text-sm leading-6 text-slate-600">{project.description}</p>
            </a>
          ))}
        </div>
      </section>

      {recentPosts.length > 0 ? (
        <section className="border-t border-slate-200 pt-12" aria-labelledby="home-recent-posts">
          <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
            <h2 id="home-recent-posts" className="text-2xl font-semibold tracking-normal text-ink">
              {copy.latestTitle}
            </h2>
            <a
              href={localizedHref("/blog/", locale)}
              className="text-sm font-medium text-slate-600 underline decoration-slate-300 underline-offset-4 transition hover:text-ink hover:decoration-slate-500"
            >
              {copy.allPosts}
            </a>
          </div>
          <BlogPostList entries={recentPosts} locale={locale} />
        </section>
      ) : null}
    </div>
  );
}
