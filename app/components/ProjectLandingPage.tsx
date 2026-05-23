import Image from "next/image";

import type {
  LocalizedText,
  ProjectCardConfig,
  ProjectGroupConfig,
  ProjectsLandingConfig
} from "../lib/content/page-config";
import type { Locale } from "../lib/site-data";
import { localizePath } from "../lib/paths";

type ProjectLandingPageProps = {
  landing: ProjectsLandingConfig;
  locale: Locale;
};

type ProjectGroupWithProjects = ProjectGroupConfig & {
  projects: ProjectCardConfig[];
};

function localizedText(value: LocalizedText, locale: Locale): string {
  return value[locale];
}

function localizedHref(href: string, locale: Locale): string {
  return href.startsWith("/") ? localizePath(href, locale) : href;
}

function resolveProjectGroups(landing: ProjectsLandingConfig): ProjectGroupWithProjects[] {
  const projectByKey = new Map(landing.projects.map((project) => [project.key, project]));

  return landing.projectGroups.map((group) => ({
    ...group,
    projects: group.projectKeys
      .map((key) => projectByKey.get(key))
      .filter((project): project is ProjectCardConfig => Boolean(project))
  }));
}

function resolveFeaturedProjects(
  landing: ProjectsLandingConfig,
  groups: ProjectGroupWithProjects[]
): ProjectCardConfig[] {
  const projectByKey = new Map(landing.projects.map((project) => [project.key, project]));
  const featured = landing.featuredProjectKeys
    .map((key) => projectByKey.get(key))
    .filter((project): project is ProjectCardConfig => Boolean(project));

  return featured.length ? featured : groups.flatMap((group) => group.projects).slice(0, 4);
}

function ProjectCard({
  project,
  locale,
  featured = false
}: {
  project: ProjectCardConfig;
  locale: Locale;
  featured?: boolean;
}) {
  const href = localizedHref(project.href, locale);

  return (
    <article
      className={[
        "group flex h-full flex-col rounded-lg border border-slate-200 bg-white transition hover:border-cyan-700/50 hover:bg-slate-50",
        featured ? "p-5" : "p-4"
      ].join(" ")}
    >
      <a href={href} className="flex min-w-0 gap-4">
        <div className="relative h-14 w-14 shrink-0 overflow-hidden rounded-md border border-slate-100 bg-slate-50">
          <Image
            src={project.image}
            alt={project.imageAlt}
            fill
            sizes="3.5rem"
            className="object-contain p-1.5"
            unoptimized
          />
        </div>
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase text-cyan-700">
            {localizedText(project.tag, locale)}
          </p>
          <h3 className="mt-1 text-lg font-semibold tracking-normal text-ink group-hover:text-cyan-800">
            {project.title}
          </h3>
        </div>
      </a>
      <p className="mt-4 flex-1 text-sm leading-6 text-slate-600">
        {localizedText(project.description, locale)}
      </p>
      {project.links.length ? (
        <div className="mt-5 flex flex-wrap gap-2 border-t border-slate-100 pt-4">
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

function ProjectGroupSection({
  group,
  locale
}: {
  group: ProjectGroupWithProjects;
  locale: Locale;
}) {
  return (
    <section className="border-t border-slate-200 py-8 first:border-t-0 first:pt-0">
      <div className="grid gap-5 lg:grid-cols-[14rem_minmax(0,1fr)]">
        <div>
          <h2 className="text-xl font-semibold tracking-normal text-ink">
            {localizedText(group.title, locale)}
          </h2>
          <p className="mt-2 text-sm leading-6 text-slate-600">{localizedText(group.intro, locale)}</p>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {group.projects.map((project) => (
            <ProjectCard key={project.key} project={project} locale={locale} />
          ))}
        </div>
      </div>
    </section>
  );
}

export function ProjectLandingPage({ landing, locale }: ProjectLandingPageProps) {
  const groups = resolveProjectGroups(landing);
  const featuredProjects = resolveFeaturedProjects(landing, groups);

  return (
    <section className="pb-16">
      <div className="border-b border-slate-200 pb-8">
        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_18rem] lg:items-end">
          <div>
            <h1 className="max-w-4xl text-4xl font-semibold tracking-normal text-ink md:text-5xl">
              {localizedText(landing.title, locale)}
            </h1>
            <p className="mt-5 max-w-3xl text-base leading-7 text-slate-600">
              {localizedText(landing.description, locale)}
            </p>
            <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-500">
              {localizedText(landing.summary, locale)}
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4">
            <div>
              <p className="text-2xl font-semibold text-ink">{landing.projectGroups.length}</p>
              <p className="mt-1 text-xs font-medium text-slate-500">
                {localizedText(landing.sectionLabels.groupCount, locale)}
              </p>
            </div>
            <div>
              <p className="text-2xl font-semibold text-ink">{landing.projects.length}</p>
              <p className="mt-1 text-xs font-medium text-slate-500">
                {localizedText(landing.sectionLabels.entryCount, locale)}
              </p>
            </div>
          </div>
        </div>
      </div>

      <section className="py-8" aria-labelledby="landing-featured">
        <h2 id="landing-featured" className="mb-4 text-2xl font-semibold tracking-normal text-ink">
          {localizedText(landing.sectionLabels.featured, locale)}
        </h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {featuredProjects.map((project) => (
            <ProjectCard key={project.key} project={project} locale={locale} featured />
          ))}
        </div>
      </section>

      <section className="border-t border-slate-200 pt-8" aria-labelledby="landing-groups">
        <h2 id="landing-groups" className="mb-1 text-2xl font-semibold tracking-normal text-ink">
          {localizedText(landing.sectionLabels.groups, locale)}
        </h2>
        {groups.map((group) => (
          <ProjectGroupSection key={group.key} group={group} locale={locale} />
        ))}
      </section>
    </section>
  );
}
