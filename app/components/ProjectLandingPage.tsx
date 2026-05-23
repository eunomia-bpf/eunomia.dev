import Image from "next/image";

import type {
  MkdocsHomeProject,
  MkdocsHomeProjectGroup,
  MkdocsLocalizedText,
  MkdocsSectionLandingPageConfig
} from "../lib/content/mkdocs-config";
import type { Locale } from "../lib/site-data";
import { localizePath } from "../lib/paths";

type ProjectLandingPageProps = {
  landing: MkdocsSectionLandingPageConfig;
  projectGroups: MkdocsHomeProjectGroup[];
  projects: MkdocsHomeProject[];
  locale: Locale;
};

type ProjectGroupWithProjects = MkdocsHomeProjectGroup & {
  projects: MkdocsHomeProject[];
};

function localizedText(value: MkdocsLocalizedText, locale: Locale): string {
  return value[locale];
}

function localizedHref(href: string, locale: Locale): string {
  return href.startsWith("/") ? localizePath(href, locale) : href;
}

function resolveProjectGroups(
  landing: MkdocsSectionLandingPageConfig,
  projectGroups: MkdocsHomeProjectGroup[],
  projects: MkdocsHomeProject[]
): ProjectGroupWithProjects[] {
  const projectByKey = new Map(projects.map((project) => [project.key, project]));
  const groups = landing.projectGroupKeys.length
    ? landing.projectGroupKeys
        .map((key) => projectGroups.find((group) => group.key === key))
        .filter((group): group is MkdocsHomeProjectGroup => Boolean(group))
    : projectGroups;

  return groups.map((group) => ({
    ...group,
    projects: group.projectKeys
      .map((key) => projectByKey.get(key))
      .filter((project): project is MkdocsHomeProject => Boolean(project))
  }));
}

function resolveFeaturedProjects(
  landing: MkdocsSectionLandingPageConfig,
  projects: MkdocsHomeProject[],
  groups: ProjectGroupWithProjects[]
): MkdocsHomeProject[] {
  const projectByKey = new Map(projects.map((project) => [project.key, project]));
  const featured = landing.featuredProjectKeys
    .map((key) => projectByKey.get(key))
    .filter((project): project is MkdocsHomeProject => Boolean(project));

  return featured.length ? featured : groups.flatMap((group) => group.projects).slice(0, 4);
}

function ProjectCard({
  project,
  locale,
  featured = false
}: {
  project: MkdocsHomeProject;
  locale: Locale;
  featured?: boolean;
}) {
  const href = localizedHref(project.href, locale);

  return (
    <article
      className={[
        "group flex h-full flex-col border border-slate-200 bg-white transition hover:border-cyan-700/50 hover:bg-slate-50",
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
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">
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
    <section className="border-t border-slate-200 pt-8 first:border-t-0 first:pt-0">
      <div className="grid gap-5 lg:grid-cols-[15rem_minmax(0,1fr)]">
        <div>
          <h2 className="text-xl font-semibold tracking-normal text-ink">{localizedText(group.title, locale)}</h2>
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

export function ProjectLandingPage({
  landing,
  projectGroups,
  projects,
  locale
}: ProjectLandingPageProps) {
  const groups = resolveProjectGroups(landing, projectGroups, projects);
  const featuredProjects = resolveFeaturedProjects(landing, projects, groups);
  const copy =
    locale === "zh"
      ? {
        featured: "重点入口",
          groups: "项目分组"
        }
      : {
          featured: "Featured entry points",
          groups: "Project groups"
        };

  return (
    <section className="pb-16">
      <div className="border-b border-slate-200 pb-8">
        <h1 className="max-w-4xl text-4xl font-semibold tracking-normal text-ink md:text-5xl">
          {localizedText(landing.title, locale)}
        </h1>
        <p className="mt-5 max-w-3xl text-base leading-7 text-slate-600">
          {localizedText(landing.description, locale)}
        </p>
      </div>

      <section className="py-8" aria-labelledby="landing-featured">
        <div className="mb-4 flex items-end justify-between gap-4">
          <h2 id="landing-featured" className="text-2xl font-semibold tracking-normal text-ink">
            {copy.featured}
          </h2>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {featuredProjects.map((project) => (
            <ProjectCard key={project.key} project={project} locale={locale} featured />
          ))}
        </div>
      </section>

      <section className="space-y-9 border-t border-slate-200 pt-8" aria-labelledby="landing-groups">
        <h2 id="landing-groups" className="text-2xl font-semibold tracking-normal text-ink">
          {copy.groups}
        </h2>
        {groups.map((group) => (
          <ProjectGroupSection key={group.key} group={group} locale={locale} />
        ))}
      </section>
    </section>
  );
}
