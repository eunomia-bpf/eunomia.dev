import Image from "next/image";

import type { ReactPageLink } from "../lib/content/types";
import type { MkdocsHomeProject } from "../lib/content/mkdocs-config";
import type { Locale } from "../lib/site-data";

type AboutLandingPageProps = {
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

function LocalLink({ link }: { link: ReactPageLink | undefined }) {
  if (!link) {
    return null;
  }

  const external = /^https?:\/\//.test(link.href);

  return (
    <a
      href={link.href}
      target={external ? "_blank" : undefined}
      rel={external ? "noopener" : undefined}
      className="inline-flex min-h-10 items-center rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-400 hover:text-ink"
    >
      {link.label}
    </a>
  );
}

export function AboutLandingPage({ locale, links, projects }: AboutLandingPageProps) {
  const linkByKey = linkMap(links);
  const logoImage = projectImage(projects, "docs");
  const copy =
    locale === "zh"
      ? {
          eyebrow: "About",
          title: "Eunomia 是围绕 eBPF、runtime extension 和 AI agent infra 的开源系统工程项目",
          description:
            "这里汇总 Eunomia 的项目体系，并组织非 eBPF 教程、研究和实验性项目。",
          tutorials: "非 eBPF 教程",
          tutorialsDescription:
            "CUDA、CUPTI 和 NVBit 内容适合作为 GPU programming、profiling 和 instrumentation 的学习入口。",
          research: "研究与实验",
          researchDescription:
            "论文、原型、roadmap 和 ideas 保留在 About 下，作为项目长期探索方向的公开记录。",
          community: "社区入口",
          communityDescription:
            "GitHub、discussion 和公开文档仍然是主要协作渠道；商业支持放在 Products。"
        }
      : {
          eyebrow: "About",
          title: "Eunomia is an open-source systems engineering effort around eBPF, runtime extension, and AI agent infrastructure",
          description:
            "This page summarizes the broader Eunomia project and organizes non-eBPF tutorials, research, and experiments.",
          tutorials: "Tutorials beyond eBPF",
          tutorialsDescription:
            "CUDA, CUPTI, and NVBit content serve as learning paths for GPU programming, profiling, and instrumentation.",
          research: "Research and experiments",
          researchDescription:
            "Papers, prototypes, roadmaps, and ideas stay under About as public records of the project's longer-term exploration.",
          community: "Community",
          communityDescription:
            "GitHub, discussions, and public documentation remain the main collaboration paths. Commercial support lives under Products."
        };

  return (
    <section className="pb-16">
      <div className="grid gap-10 border-b border-slate-200 pb-12 lg:grid-cols-[minmax(0,1fr)_22rem] lg:items-center">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">{copy.eyebrow}</p>
          <h1 className="mt-4 max-w-5xl text-4xl font-semibold tracking-normal text-ink md:text-5xl">
            {copy.title}
          </h1>
          <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-600">{copy.description}</p>
        </div>
        {logoImage ? (
          <div className="relative min-h-64 overflow-hidden border border-slate-200 bg-slate-50">
            <Image src={logoImage} alt="Eunomia" fill sizes="22rem" className="object-contain p-12" unoptimized />
          </div>
        ) : null}
      </div>

      <div className="grid gap-8 py-12 lg:grid-cols-[18rem_minmax(0,1fr)]">
        <div>
          <h2 className="text-2xl font-semibold tracking-normal text-ink">{copy.tutorials}</h2>
          <p className="mt-3 text-sm leading-6 text-slate-600">{copy.tutorialsDescription}</p>
        </div>
        <div className="grid gap-3 md:grid-cols-3">
          <article className="border border-slate-200 bg-white p-5">
            <h3 className="text-lg font-semibold tracking-normal text-ink">CUDA Tutorial</h3>
            <p className="mt-3 text-sm leading-6 text-slate-600">
              {locale === "zh" ? "CUDA 基础、PTX、GPU 架构和高级定制。" : "CUDA basics, PTX, GPU architecture, and advanced customization."}
            </p>
            <div className="mt-5">
              <LocalLink link={linkByKey.get("cuda-tutorial")} />
            </div>
          </article>
          <article className="border border-slate-200 bg-white p-5">
            <h3 className="text-lg font-semibold tracking-normal text-ink">CUPTI Tutorial</h3>
            <p className="mt-3 text-sm leading-6 text-slate-600">
              {locale === "zh" ? "CUDA profiling、activity tracing、sampling 和 injection 示例。" : "CUDA profiling, activity tracing, sampling, and injection examples."}
            </p>
            <div className="mt-5">
              <LocalLink link={linkByKey.get("cupti-tutorial")} />
            </div>
          </article>
          <article className="border border-slate-200 bg-white p-5">
            <h3 className="text-lg font-semibold tracking-normal text-ink">NVBit Tutorial</h3>
            <p className="mt-3 text-sm leading-6 text-slate-600">
              {locale === "zh" ? "GPU binary instrumentation 的工具和内部机制。" : "Tools and internals for GPU binary instrumentation."}
            </p>
            <div className="mt-5">
              <LocalLink link={linkByKey.get("nvbit-tutorial")} />
            </div>
          </article>
        </div>
      </div>

      <div className="grid gap-8 border-t border-slate-200 py-12 lg:grid-cols-2">
        <article>
          <h2 className="text-2xl font-semibold tracking-normal text-ink">{copy.research}</h2>
          <p className="mt-3 text-sm leading-6 text-slate-600">{copy.researchDescription}</p>
          <div className="mt-5 flex flex-wrap gap-3">
            <LocalLink link={linkByKey.get("papers")} />
            <LocalLink link={linkByKey.get("ideas")} />
            <LocalLink link={linkByKey.get("usecases")} />
          </div>
        </article>
        <article>
          <h2 className="text-2xl font-semibold tracking-normal text-ink">{copy.community}</h2>
          <p className="mt-3 text-sm leading-6 text-slate-600">{copy.communityDescription}</p>
          <div className="mt-5 flex flex-wrap gap-3">
            <LocalLink link={linkByKey.get("github")} />
            <LocalLink link={linkByKey.get("discussions")} />
            <LocalLink link={linkByKey.get("products")} />
          </div>
        </article>
      </div>
    </section>
  );
}
