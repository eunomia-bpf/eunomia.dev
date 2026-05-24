import Image from "next/image";

import type { MkdocsHomeConfig, MkdocsHomeProject, MkdocsHomeProjectGroup } from "../lib/content/mkdocs-config";
import type { BlogEntry } from "../lib/content/types";
import { localizePath } from "../lib/paths";
import type { Locale } from "../lib/site-data";
import { BlogPostList } from "./BlogPostList";

type HomePageLandingProps = {
  locale: Locale;
  recentPosts: BlogEntry[];
  home: MkdocsHomeConfig;
};

function localizedHref(href: string, locale: Locale): string {
  return href.startsWith("/") ? localizePath(href, locale) : href;
}

function localizedText(value: Record<Locale, string>, locale: Locale): string {
  return value[locale];
}

function ProjectCard({ project, locale }: { project: MkdocsHomeProject; locale: Locale }) {
  const href = localizedHref(project.href, locale);

  return (
    <article className="group rounded-lg border border-slate-200 bg-white p-4 transition hover:border-cyan-700/40 hover:bg-slate-50">
      <a href={href} className="flex min-w-0 gap-3">
        {project.image ? (
          <div className="relative h-12 w-12 shrink-0 overflow-hidden rounded-md border border-slate-100 bg-slate-50">
            <Image
              src={project.image}
              alt={project.imageAlt}
              fill
              sizes="3rem"
              className="object-contain p-1.5"
              unoptimized
            />
          </div>
        ) : null}
        <div className="min-w-0 flex-1">
          <p className="text-xs font-semibold uppercase tracking-normal text-cyan-700">
            {localizedText(project.tag, locale)}
          </p>
          <h3 className="mt-1 text-base font-semibold tracking-normal text-ink group-hover:text-cyan-800">
            {project.title}
          </h3>
          <p className="mt-2 line-clamp-2 text-sm leading-6 text-slate-600">
            {localizedText(project.description, locale)}
          </p>
        </div>
      </a>
      {project.links.length ? (
        <div className="mt-4 flex flex-wrap gap-2 border-t border-slate-100 pt-3">
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

function groupProjects(projects: MkdocsHomeProject[], groups: MkdocsHomeProjectGroup[]) {
  const byKey = new Map(projects.map((project) => [project.key, project]));
  const assignedKeys = new Set(groups.flatMap((group) => group.projectKeys));
  const remainingProjects = projects.filter((project) => !assignedKeys.has(project.key));

  return groups.map((group, index) => ({
    ...group,
    projects: [
      ...group.projectKeys.map((key) => byKey.get(key)).filter((project): project is MkdocsHomeProject => Boolean(project)),
      ...(index === groups.length - 1 ? remainingProjects : [])
    ]
  }));
}

function CapabilitySection({ home, locale }: { home: MkdocsHomeConfig; locale: Locale }) {
  return (
    <section className="border-b border-slate-200 pb-12" aria-labelledby="home-capabilities">
      <div className="mb-6">
        <h2 id="home-capabilities" className="text-2xl font-semibold tracking-normal text-ink">
          {localizedText(home.capabilitiesTitle, locale)}
        </h2>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
          {localizedText(home.capabilitiesIntro, locale)}
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        {home.capabilities.map((item) => (
          <article key={item.key} className="rounded-lg border border-slate-200 bg-white p-5">
            <p className="text-xs font-semibold uppercase tracking-normal text-cyan-700">
              {localizedText(item.eyebrow, locale)}
            </p>
            <h3 className="mt-3 text-lg font-semibold tracking-normal text-ink">
              {localizedText(item.title, locale)}
            </h3>
            <p className="mt-3 text-sm leading-6 text-slate-600">{localizedText(item.description, locale)}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

export function HomePageHero({ home, locale }: { home: MkdocsHomeConfig; locale: Locale }) {
  return (
    <section className="relative isolate overflow-hidden border-b border-slate-200 bg-white text-ink">
      <div
        aria-hidden="true"
        className="absolute inset-0 -z-10 bg-[linear-gradient(to_right,rgba(15,23,42,0.06)_1px,transparent_1px),linear-gradient(to_bottom,rgba(15,23,42,0.06)_1px,transparent_1px)] bg-[size:24px_24px]"
      />
      <div
        aria-hidden="true"
        className="absolute inset-x-0 top-0 -z-10 h-72 bg-gradient-to-b from-cyan-100/70 via-cyan-50/40 to-white"
      />
      <div className="relative mx-auto max-w-[82rem] px-4 py-20 sm:px-6 lg:px-8 lg:py-28">
        <div className="mx-auto max-w-4xl text-center">
          <p className="inline-flex items-center rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-cyan-800 shadow-sm backdrop-blur">
            {localizedText(home.hero.kicker, locale)}
          </p>
          <h1 className="mt-6 font-serif text-5xl font-normal tracking-normal text-slate-950 sm:text-6xl lg:text-7xl">
            {localizedText(home.hero.title, locale)}
          </h1>
          <p className="mx-auto mt-5 max-w-3xl text-xl leading-8 text-slate-600 md:text-2xl md:leading-9">
            {localizedText(home.hero.summary, locale)}
          </p>
          <div className="mt-9 flex flex-wrap justify-center gap-3">
            <a
              href={localizedHref(home.hero.primaryHref, locale)}
              className="inline-flex min-h-11 items-center rounded-md bg-slate-950 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800"
            >
              {localizedText(home.hero.primaryCta, locale)}
            </a>
            <a
              href={localizedHref(home.hero.secondaryHref, locale)}
              className="inline-flex min-h-11 items-center rounded-md border border-slate-300 bg-white/80 px-5 py-2.5 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-slate-400 hover:text-ink"
            >
              {localizedText(home.hero.secondaryCta, locale)}
            </a>
            {home.hero.tertiaryCta && home.hero.tertiaryHref ? (
              <a
                href={localizedHref(home.hero.tertiaryHref, locale)}
                className="inline-flex min-h-11 items-center rounded-md border border-slate-300 bg-white/80 px-5 py-2.5 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-slate-400 hover:text-ink"
              >
                {localizedText(home.hero.tertiaryCta, locale)}
              </a>
            ) : null}
          </div>
        </div>
      </div>
    </section>
  );
}

export function HomePageLanding({ locale, recentPosts, home }: HomePageLandingProps) {
  const projectGroups = groupProjects(home.projects, home.projectGroups);

  return (
    <div className="pb-16">
      <CapabilitySection home={home} locale={locale} />

      <section className="border-b border-slate-200 py-12" aria-labelledby="home-projects">
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
        <div className="space-y-8">
          {projectGroups.map((group) => (
            <section
              key={group.key}
              className="border-t border-slate-200 pt-6 first:border-t-0 first:pt-0"
            >
              <div className="grid gap-4 lg:grid-cols-[14rem_minmax(0,1fr)] lg:items-start">
                <div>
                  <h3 className="text-lg font-semibold tracking-normal text-ink">
                    {localizedText(group.title, locale)}
                  </h3>
                  <p className="mt-2 text-sm leading-6 text-slate-600">
                    {localizedText(group.intro, locale)}
                  </p>
                </div>
                <div className="-mx-4 overflow-x-auto px-4 pb-3 lg:mx-0 lg:px-0">
                  <div className="flex gap-3">
                    {group.projects.map((project) => (
                      <div key={project.key} className="w-[19rem] flex-none sm:w-[21rem] lg:w-[23rem]">
                        <ProjectCard project={project} locale={locale} />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </section>
          ))}
        </div>
      </section>

      {recentPosts.length > 0 ? (
        <section className="pt-12" aria-labelledby="home-recent-posts">
          <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
            <h2 id="home-recent-posts" className="text-2xl font-semibold tracking-normal text-ink">
              {localizedText(home.latestTitle, locale)}
            </h2>
            <a
              href={localizedHref("/blog/", locale)}
              className="text-sm font-medium text-slate-600 underline decoration-slate-300 underline-offset-4 transition hover:text-ink hover:decoration-slate-500"
            >
              {localizedText(home.allPostsLabel, locale)}
            </a>
          </div>
          <BlogPostList entries={recentPosts} locale={locale} />
        </section>
      ) : null}
    </div>
  );
}
