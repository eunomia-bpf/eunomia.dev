import Image from "next/image";

import type { MkdocsHomeConfig, MkdocsHomeProject } from "../lib/content/mkdocs-config";
import type { BlogEntry } from "../lib/content/types";
import { localizePath } from "../lib/paths";
import type { Locale } from "../lib/site-data";
import { BlogPostList } from "./BlogPostList";

type HomePageLandingProps = {
  locale: Locale;
  recentPosts: BlogEntry[];
  home: MkdocsHomeConfig;
};

type HomeCopy = {
  kicker: string;
  title: string;
  summary: string;
  primaryCta: string;
  secondaryCta: string;
  secondaryHref: string;
  latestTitle: string;
  allPosts: string;
};

const copyByLocale: Record<Locale, HomeCopy> = {
  en: {
    kicker: "Open systems research lab",
    title: "Eunomia",
    summary:
      "Open-source eBPF systems infrastructure for runtime extension, tracing, packaging, and reproducible research artifacts.",
    primaryCta: "Explore docs",
    secondaryCta: "View GitHub",
    secondaryHref: "https://github.com/eunomia-bpf",
    latestTitle: "Latest writing",
    allPosts: "All posts"
  },
  zh: {
    kicker: "开源系统研究实验室",
    title: "Eunomia",
    summary:
      "面向 eBPF 的开源系统基础设施，覆盖 runtime 扩展、tracing、程序打包和可复现研究产物。",
    primaryCta: "浏览文档",
    secondaryCta: "查看 GitHub",
    secondaryHref: "https://github.com/eunomia-bpf",
    latestTitle: "最新文章",
    allPosts: "全部文章"
  }
};

function localizedHref(href: string, locale: Locale): string {
  return href.startsWith("/") ? localizePath(href, locale) : href;
}

function localizedText(value: Record<Locale, string>, locale: Locale): string {
  return value[locale] || value.en || value.zh;
}

function ProjectCard({ project, locale }: { project: MkdocsHomeProject; locale: Locale }) {
  const href = localizedHref(project.href, locale);

  return (
    <article className="group flex h-[17.75rem] flex-col overflow-hidden rounded-lg border border-slate-200 bg-white transition hover:border-cyan-700/40 hover:bg-slate-50">
      <a href={href} className="flex min-w-0 flex-1 flex-col">
        {project.image ? (
          <div className="relative h-20 shrink-0 border-b border-slate-100 bg-slate-50">
            <Image
              src={project.image}
              alt={project.imageAlt || project.title}
              fill
              sizes="18rem"
              className="object-contain p-3"
              unoptimized
            />
          </div>
        ) : null}
        <div className="min-h-0 flex-1 p-4">
          <p className="text-xs font-semibold uppercase tracking-normal text-cyan-700">
            {localizedText(project.tag, locale)}
          </p>
          <h3 className="mt-2 text-lg font-semibold tracking-normal text-ink group-hover:text-cyan-800">
            {project.title}
          </h3>
          <p className="mt-2 line-clamp-2 text-sm leading-6 text-slate-600">
            {localizedText(project.description, locale)}
          </p>
        </div>
      </a>
      {project.links.length ? (
        <div className="flex shrink-0 flex-wrap gap-2 border-t border-slate-100 px-4 pb-4 pt-3">
          {project.links.map((link) => (
            <a
              key={`${project.key}:${link.href}:${localizedText(link.label, locale)}`}
              href={localizedHref(link.href, locale)}
              className="inline-flex min-h-8 items-center rounded-md border border-slate-200 px-2.5 text-xs font-semibold text-slate-600 transition hover:border-cyan-700/40 hover:text-cyan-800"
            >
              {localizedText(link.label, locale)}
            </a>
          ))}
        </div>
      ) : null}
    </article>
  );
}

export function HomePageHero({ locale }: { locale: Locale }) {
  const copy = copyByLocale[locale];

  return (
    <section className="relative isolate overflow-hidden border-b border-slate-200 bg-[#07111f] text-white">
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
      <div className="absolute inset-0 bg-[#07111f]/80" />
      <div className="relative mx-auto max-w-[82rem] px-4 py-14 sm:px-6 lg:px-8 lg:py-20">
        <div className="max-w-3xl">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-200">{copy.kicker}</p>
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
              href={localizedHref(copy.secondaryHref, locale)}
              className="inline-flex min-h-11 items-center rounded-lg border border-white/35 px-5 py-2.5 text-sm font-semibold text-white transition hover:border-white hover:bg-white/10"
            >
              {copy.secondaryCta}
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}

export function HomePageLanding({ locale, recentPosts, home }: HomePageLandingProps) {
  const copy = copyByLocale[locale];

  return (
    <div className="pb-16">
      <section className="border-b border-slate-200 pb-12" aria-labelledby="home-projects">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 id="home-projects" className="text-2xl font-semibold tracking-normal text-ink">
              {localizedText(home.projectsTitle, locale)}
            </h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
              {localizedText(home.projectsIntro, locale)}
            </p>
          </div>
        </div>
        <div className="overflow-x-auto pb-3">
          <div className="grid w-max grid-flow-col grid-rows-2 gap-4 pr-4 [grid-auto-columns:minmax(17rem,18.5rem)] sm:[grid-auto-columns:20rem] lg:[grid-auto-columns:21rem]">
            {home.projects.map((project) => (
              <ProjectCard key={project.key} project={project} locale={locale} />
            ))}
          </div>
        </div>
      </section>

      {recentPosts.length > 0 ? (
        <section className="pt-12" aria-labelledby="home-recent-posts">
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
