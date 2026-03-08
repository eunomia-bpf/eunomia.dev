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
    <section className="border-b border-slate-200/80 bg-white/70 pb-12">
      <section className="mx-auto grid max-w-[94rem] gap-8 px-4 pb-8 pt-8 sm:px-6 lg:grid-cols-[minmax(0,1.2fr)_19rem] lg:px-8">
        <div className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm md:p-10">
          <span className="inline-flex rounded-full bg-slate-100 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-600">
            {copy.badge}
          </span>
          <h1 className="mt-5 max-w-4xl text-4xl font-semibold leading-tight tracking-tight text-ink md:text-5xl">
            {copy.headline}
          </h1>
          <p className="mt-5 max-w-3xl text-base leading-8 text-slate-600 md:text-lg">{copy.body}</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href={page.cards[0]?.href ?? localizePath("/tutorials/", locale)}
              className="inline-flex rounded-lg bg-ink px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
            >
              {copy.primaryCta}
            </a>
            <a
              href={page.spotlight.href}
              className="inline-flex rounded-lg border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:text-ink"
            >
              {copy.secondaryCta}
            </a>
            <a
              href={siteConfig.repoUrl}
              className="inline-flex rounded-lg border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:text-ink"
            >
              {copy.tertiaryCta}
            </a>
          </div>
          <div className="mt-10 grid gap-4 md:grid-cols-3">
            {page.stats.map((stat) => (
              <div key={stat.label} className="rounded-2xl border border-slate-200 bg-slate-50/90 p-5">
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">{stat.label}</p>
                <p className="mt-2 text-3xl font-semibold tracking-tight text-ink">{stat.value}</p>
                <p className="mt-2 text-sm leading-6 text-slate-600">{stat.detail}</p>
              </div>
            ))}
          </div>
        </div>

        <aside className="space-y-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{copy.spotlightLabel}</p>
            <h2 className="mt-3 text-2xl font-semibold tracking-tight text-ink">{page.spotlight.title}</h2>
            <p className="mt-3 text-sm leading-7 text-slate-600">{page.spotlight.description}</p>
            <a href={page.spotlight.href} className="mt-5 inline-flex text-sm font-semibold text-slate-900">
              {page.spotlight.badge}
            </a>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{copy.signalLabel}</p>
            <h2 className="mt-3 text-xl font-semibold tracking-tight text-ink">{copy.signalTitle}</h2>
            <ul className="mt-4 space-y-3 text-sm leading-7 text-slate-600">
              {copy.signals.map((signal) => (
                <li key={signal} className="rounded-xl bg-slate-50 px-4 py-3">
                  {signal}
                </li>
              ))}
            </ul>
          </div>
        </aside>
      </section>

      <section className="mx-auto max-w-[94rem] px-4 pb-8 sm:px-6 lg:px-8">
        <div className="mb-5">
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{copy.tracksLabel}</p>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight text-ink">{page.title}</h2>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {page.cards.map((card) => (
            <a
              key={card.href}
              href={card.href}
              className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition hover:border-slate-300"
            >
              {card.badge ? (
                <span className="inline-flex rounded-full bg-slate-100 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-600">
                  {card.badge}
                </span>
              ) : null}
              <h3 className="mt-4 text-xl font-semibold tracking-tight text-ink">{card.title}</h3>
              <p className="mt-3 leading-7 text-slate-600">{card.description}</p>
              <span className="mt-5 inline-flex text-sm font-semibold text-slate-900">{copy.openCard}</span>
            </a>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-[94rem] px-4 pb-8 sm:px-6 lg:px-8">
        <div className="mb-5">
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{copy.exploreLabel}</p>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {copy.moreLinks.map((card) => (
            <a
              key={card.href}
              href={card.href}
              className="rounded-2xl border border-slate-200 bg-white p-6 transition hover:border-slate-300"
            >
              <h3 className="text-lg font-semibold tracking-tight text-ink">{card.title}</h3>
              <p className="mt-3 text-sm leading-6 text-slate-600">{card.description}</p>
            </a>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-[94rem] px-4 sm:px-6 lg:px-8">
        <article className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm md:p-10">
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
