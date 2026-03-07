import type { HomePageData } from "../lib/page-factories";
import { localizePath } from "../lib/paths";
import { type Locale, siteConfig } from "../lib/site-data";
import { homeLandingCopyByLocale } from "../lib/ui-copy";
import { PageFooter } from "./PageFooter";

type HomeLandingProps = {
  locale: Locale;
  page: HomePageData;
};

export function HomeLanding({ locale, page }: HomeLandingProps) {
  const copy = homeLandingCopyByLocale[locale];

  return (
    <section className="pb-16">
      <section className="mx-auto grid max-w-6xl gap-8 px-5 pb-10 pt-14 xl:grid-cols-[minmax(0,1.35fr)_23rem]">
        <div className="rounded-[2.5rem] border border-white/70 bg-white/85 p-8 shadow-panel md:p-12">
          <span className="inline-flex rounded-full bg-ink px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-white">
            {copy.badge}
          </span>
          <h1 className="mt-6 max-w-4xl font-serif text-5xl leading-[1.02] tracking-tight text-ink md:text-7xl">
            {copy.headline}
          </h1>
          <p className="mt-6 max-w-3xl text-lg leading-8 text-slate-600">{copy.body}</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href={page.cards[0]?.href ?? localizePath("/tutorials/", locale)}
              className="inline-flex rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white transition hover:bg-azure"
            >
              {copy.primaryCta}
            </a>
            <a
              href={page.spotlight.href}
              className="inline-flex rounded-full border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-azure hover:text-azure"
            >
              {copy.secondaryCta}
            </a>
            <a
              href={siteConfig.repoUrl}
              className="inline-flex rounded-full border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-azure hover:text-azure"
            >
              {copy.tertiaryCta}
            </a>
          </div>
          <div className="mt-10 grid gap-4 md:grid-cols-3">
            {page.stats.map((stat) => (
              <div key={stat.label} className="rounded-[1.5rem] border border-slate-200 bg-slate-50/80 p-5">
                <p className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">{stat.label}</p>
                <p className="mt-3 text-3xl font-semibold tracking-tight text-ink">{stat.value}</p>
                <p className="mt-2 text-sm leading-6 text-slate-600">{stat.detail}</p>
              </div>
            ))}
          </div>
        </div>

        <aside className="space-y-6">
          <div className="rounded-[2.25rem] border border-ink/10 bg-ink p-7 text-white shadow-panel">
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-amber-200">{copy.spotlightLabel}</p>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight">{page.spotlight.title}</h2>
            <p className="mt-4 text-sm leading-7 text-slate-200">{page.spotlight.description}</p>
            <a
              href={page.spotlight.href}
              className="mt-6 inline-flex rounded-full bg-white px-4 py-2 text-sm font-semibold text-ink transition hover:bg-amber-100"
            >
              {page.spotlight.badge}
            </a>
          </div>
          <div className="rounded-[2.25rem] border border-white/70 bg-white/90 p-7 shadow-panel">
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-azure">{copy.signalLabel}</p>
            <h2 className="mt-4 text-2xl font-semibold tracking-tight text-ink">{copy.signalTitle}</h2>
            <ul className="mt-5 space-y-3 text-sm leading-7 text-slate-600">
              {copy.signals.map((signal) => (
                <li key={signal} className="rounded-2xl bg-slate-50 px-4 py-3">
                  {signal}
                </li>
              ))}
            </ul>
          </div>
        </aside>
      </section>

      <section className="mx-auto max-w-6xl px-5 pb-10">
        <div className="mb-6 flex items-center justify-between gap-4">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.22em] text-slate-500">{copy.tracksLabel}</p>
            <h2 className="mt-2 text-3xl font-semibold tracking-tight text-ink">{page.title}</h2>
          </div>
        </div>
        <div className="grid gap-6 md:grid-cols-3">
          {page.cards.map((card) => (
            <a
              key={card.href}
              href={card.href}
              className="group rounded-[2rem] border border-white/70 bg-white/90 p-7 shadow-panel transition duration-300 hover:-translate-y-1 hover:border-azure/30"
            >
              {card.badge ? (
                <span className="inline-flex rounded-full bg-mist px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-azure">
                  {card.badge}
                </span>
              ) : null}
              <h3 className="mt-5 text-2xl font-semibold tracking-tight text-ink">{card.title}</h3>
              <p className="mt-4 leading-7 text-slate-600">{card.description}</p>
              <span className="mt-6 inline-flex text-sm font-semibold text-azure transition group-hover:translate-x-1">
                {copy.openCard}
              </span>
            </a>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-5 pb-10">
        <div className="mb-6">
          <p className="text-sm font-semibold uppercase tracking-[0.22em] text-slate-500">{copy.exploreLabel}</p>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {copy.moreLinks.map((card) => (
            <a
              key={card.href}
              href={card.href}
              className="rounded-[1.75rem] border border-slate-200 bg-white/85 p-6 transition hover:border-azure hover:shadow-sm"
            >
              <h3 className="text-xl font-semibold tracking-tight text-ink">{card.title}</h3>
              <p className="mt-3 text-sm leading-6 text-slate-600">{card.description}</p>
            </a>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-5">
        <article className="rounded-[2rem] border border-white/70 bg-white/90 p-8 shadow-panel md:p-10">
          <PageFooter
            locale={locale}
            title={page.title}
            path={page.path}
            sourceHref={page.sourcePath}
            metadata={page.metadata}
          />
        </article>
      </section>
    </section>
  );
}
