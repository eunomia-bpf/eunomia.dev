import Image from "next/image";

import type {
  AiEbpfLandingConfig,
  LocalizedText,
  ProjectCardConfig
} from "../lib/content/page-config";
import { localizePath } from "../lib/paths";
import type { Locale } from "../lib/site-data";

type AiEbpfLandingPageProps = {
  landing: AiEbpfLandingConfig;
  locale: Locale;
};

function localizedText(value: LocalizedText, locale: Locale): string {
  return value[locale];
}

function localizedHref(href: string, locale: Locale): string {
  return href.startsWith("/") ? localizePath(href, locale) : href;
}

function ProjectCard({ project, locale }: { project: ProjectCardConfig; locale: Locale }) {
  return (
    <article className="group flex h-full flex-col rounded-lg border border-slate-200 bg-white p-4 transition hover:border-cyan-700/50 hover:bg-slate-50">
      <a href={localizedHref(project.href, locale)} className="flex min-w-0 gap-4">
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

export function AiEbpfLandingPage({ landing, locale }: AiEbpfLandingPageProps) {
  return (
    <section className="pb-16">
      <section className="relative isolate -mx-4 overflow-hidden border-b border-slate-200 bg-[#07111f] px-4 py-10 text-white sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8">
        <Image
          src={landing.heroImage}
          alt=""
          aria-hidden="true"
          fill
          sizes="100vw"
          className="object-cover opacity-20"
          unoptimized
        />
        <div className="absolute inset-0 bg-[#07111f]/80" />
        <div className="relative max-w-4xl">
          <h1 className="text-4xl font-semibold tracking-normal text-white md:text-5xl">
            {localizedText(landing.title, locale)}
          </h1>
          <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-200">
            {localizedText(landing.description, locale)}
          </p>
          <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-300">
            {localizedText(landing.summary, locale)}
          </p>
        </div>
      </section>

      <section className="py-9" aria-labelledby="ai-directions">
        <h2 id="ai-directions" className="mb-4 text-2xl font-semibold tracking-normal text-ink">
          {localizedText(landing.sectionLabels.directions, locale)}
        </h2>
        <div className="grid gap-4 lg:grid-cols-2">
          {landing.directions.map((direction) => (
            <article key={direction.key} className="rounded-lg border border-slate-200 bg-white p-5">
              <p className="text-xs font-semibold uppercase text-cyan-700">
                {localizedText(direction.eyebrow, locale)}
              </p>
              <h3 className="mt-3 text-xl font-semibold tracking-normal text-ink">
                {localizedText(direction.title, locale)}
              </h3>
              <p className="mt-3 text-sm leading-6 text-slate-600">
                {localizedText(direction.description, locale)}
              </p>
              <ul className="mt-5 space-y-2 border-t border-slate-100 pt-4">
                {direction.points.map((point) => (
                  <li key={localizedText(point.label, locale)} className="flex gap-2 text-sm leading-6 text-slate-700">
                    <span aria-hidden="true" className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-700" />
                    <span>{localizedText(point.label, locale)}</span>
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </section>

      <section className="border-t border-slate-200 py-9" aria-labelledby="ai-projects">
        <h2 id="ai-projects" className="mb-4 text-2xl font-semibold tracking-normal text-ink">
          {localizedText(landing.sectionLabels.projects, locale)}
        </h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {landing.featuredProjects.map((project) => (
            <ProjectCard key={project.key} project={project} locale={locale} />
          ))}
        </div>
      </section>

      <section className="border-t border-slate-200 py-9" aria-labelledby="ai-use-cases">
        <h2 id="ai-use-cases" className="mb-4 text-2xl font-semibold tracking-normal text-ink">
          {localizedText(landing.sectionLabels.useCases, locale)}
        </h2>
        <div className="grid gap-5 lg:grid-cols-2">
          {landing.useCaseGroups.map((group) => (
            <article key={group.key} className="rounded-lg border border-slate-200 bg-white p-5">
              <p className="text-xs font-semibold uppercase text-slate-500">
                {localizedText(group.eyebrow, locale)}
              </p>
              <h3 className="mt-2 text-xl font-semibold tracking-normal text-ink">
                {localizedText(group.title, locale)}
              </h3>
              <p className="mt-3 text-sm leading-6 text-slate-600">
                {localizedText(group.description, locale)}
              </p>
              <div className="mt-5 space-y-3 border-t border-slate-100 pt-4">
                {group.items.map((item) => {
                  const title = localizedText(item.title, locale);
                  const content = (
                    <>
                      <p className="text-sm font-semibold text-ink">{title}</p>
                      <p className="mt-1 text-sm leading-6 text-slate-600">
                        {localizedText(item.description, locale)}
                      </p>
                    </>
                  );

                  return item.href ? (
                    <a
                      key={title}
                      href={localizedHref(item.href, locale)}
                      className="block rounded-md border border-slate-100 p-3 transition hover:border-cyan-700/40 hover:bg-slate-50"
                    >
                      {content}
                    </a>
                  ) : (
                    <div key={title} className="rounded-md border border-slate-100 p-3">
                      {content}
                    </div>
                  );
                })}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="border-t border-slate-200 pt-8" aria-labelledby="ai-references">
        <h2 id="ai-references" className="mb-4 text-2xl font-semibold tracking-normal text-ink">
          {localizedText(landing.sectionLabels.references, locale)}
        </h2>
        <div className="flex flex-wrap gap-2">
          {landing.references.map((link) => (
            <a
              key={`${link.href}:${localizedText(link.label, locale)}`}
              href={localizedHref(link.href, locale)}
              className="inline-flex min-h-9 items-center rounded-md border border-slate-200 px-3 text-sm font-semibold text-slate-600 transition hover:border-cyan-700/40 hover:text-cyan-800"
            >
              {localizedText(link.label, locale)}
            </a>
          ))}
        </div>
      </section>
    </section>
  );
}
