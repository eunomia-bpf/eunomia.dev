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
    <section className="border-b border-slate-200 bg-white pb-12">
      <section className="mx-auto grid max-w-[94rem] gap-8 px-4 pb-8 pt-8 sm:px-6 lg:grid-cols-[minmax(0,1.2fr)_19rem] lg:px-8">
        <div className="border border-slate-200 bg-white p-8 md:p-10">
          <span className="inline-flex text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
            {copy.badge}
          </span>
          <h1 className="mt-5 max-w-4xl text-4xl font-semibold leading-tight tracking-tight text-ink md:text-5xl">
            {copy.headline}
          </h1>
          <p className="mt-5 max-w-3xl text-base leading-8 text-slate-600 md:text-lg">{copy.body}</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href={page.cards[0]?.href ?? localizePath("/tutorials/", locale)}
              className="inline-flex rounded-lg border border-slate-900 bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
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
          <dl className="mt-10 grid gap-4 border-t border-slate-200 pt-6 md:grid-cols-3">
            {page.stats.map((stat) => (
              <div key={stat.label}>
                <dt className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">{stat.label}</dt>
                <dd className="mt-2 text-3xl font-semibold tracking-tight text-ink">{stat.value}</dd>
                <dd className="mt-2 text-sm leading-6 text-slate-600">{stat.detail}</dd>
              </div>
            ))}
          </dl>
        </div>

        <aside className="space-y-4">
          <div className="border border-slate-200 bg-white p-6">
            <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{copy.spotlightLabel}</p>
            <h2 className="mt-3 text-2xl font-semibold tracking-tight text-ink">{page.spotlight.title}</h2>
            <p className="mt-3 text-sm leading-7 text-slate-600">{page.spotlight.description}</p>
            <a href={page.spotlight.href} className="mt-5 inline-flex text-sm font-semibold text-slate-900 underline underline-offset-4">
              {page.spotlight.badge}
            </a>
          </div>
          <div className="border border-slate-200 bg-white p-6">
            <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{copy.signalLabel}</p>
            <h2 className="mt-3 text-xl font-semibold tracking-tight text-ink">{copy.signalTitle}</h2>
            <ul className="mt-4 space-y-3 text-sm leading-7 text-slate-600">
              {copy.signals.map((signal) => (
                <li key={signal}>{signal}</li>
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
        <div className="overflow-hidden border border-slate-200 bg-white">
          {page.cards.map((card, index) => (
            <a
              key={card.href}
              href={card.href}
              className={`block px-6 py-5 transition hover:bg-slate-50 ${index > 0 ? "border-t border-slate-200" : ""}`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <h3 className="text-lg font-semibold tracking-tight text-ink">{card.title}</h3>
                  <p className="mt-2 leading-7 text-slate-600">{card.description}</p>
                  <span className="mt-4 inline-flex text-sm font-semibold text-slate-900">{copy.openCard}</span>
                </div>
                {card.badge ? (
                  <span className="rounded-md bg-slate-100 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-600">
                    {card.badge}
                  </span>
                ) : null}
              </div>
            </a>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-[94rem] px-4 pb-8 sm:px-6 lg:px-8">
        <div className="mb-5">
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">{copy.exploreLabel}</p>
        </div>
        <div className="overflow-hidden border border-slate-200 bg-white">
          {copy.moreLinks.map((card, index) => (
            <a
              key={card.href}
              href={card.href}
              className={`block px-6 py-5 transition hover:bg-slate-50 ${index > 0 ? "border-t border-slate-200" : ""}`}
            >
              <h3 className="text-lg font-semibold tracking-tight text-ink">{card.title}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-600">{card.description}</p>
            </a>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-[94rem] px-4 sm:px-6 lg:px-8">
        <article className="border border-slate-200 bg-white p-8 md:p-10">
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
